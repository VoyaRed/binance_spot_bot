// live_executor.js - project neptune live execution engine (Multi-Pair FCFS)
require('dotenv').config();
const http = require('http');
const ccxt = require('ccxt');
const ort = require('onnxruntime-node');
const { createClient } = require('@supabase/supabase-js');
const cron = require('node-cron');

// --- render free tier dummy server ---
const PORT = process.env.PORT || 3000;
const server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('project neptune engine is active.\n');
});
server.listen(PORT, () => {
    logCyan(`web service bound to port ${PORT} to maintain render lifecycle.`);
});

// --- supabase telemetry ---
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const TIMEFRAME = '15m';
const PAPER_TRADING = process.env.PAPER_TRADING === 'true';

const exchange = new ccxt.binanceus({
    apiKey: process.env.BINANCEUS_API_KEY,
    secret: process.env.BINANCEUS_API_SECRET,
    enableRateLimit: true
});

const logCyan = (msg) => console.log(`\x1b[36m${msg.toLowerCase()}\x1b[0m`);

// --- MULTI-PAIR FLEET CONFIGURATION ---
const FLEET = [
    { ticker: 'ETH/USDT', modelFile: 'veto_engine_spot_15m_eth.onnx', threshold: 0.4807 },
    { ticker: 'BTC/USDT', modelFile: 'veto_engine_spot_15m_btc.onnx', threshold: 0.4735 }, 
    { ticker: 'DOGE/USDT', modelFile: 'veto_engine_spot_15m_doge.onnx', threshold: 0.4800 } // Update after training
];

// --- SYNCHRONIZED MATH CLOSURES ---
const settings = {
    ema_fast_period: 9, ema_slow_period: 21, macro_ema_fast: 50, macro_ema_slow: 200, htf_filter_period: 160  
};

const riskSettings = {
    atrStopMultiplier: 2.0, atrTargetMultiplier: 2.0, forwardHorizonCandles: 12,
    priceImpactPerc: 0.0002, takerFeePerc: 0.00019, minVolFeeMultiplier: 6.0    
};

const calculateEMAArray = (data, period) => {
    if (data.length < period) return [];
    const k = 2 / (period + 1);
    let emaArray = [data[0]];
    for (let i = 1; i < data.length; i++) {
        emaArray.push((data[i] * k) + (emaArray[i - 1] * (1 - k)));
    }
    return emaArray;
};

const calcStdDev = (arr, mean) => {
    if (arr.length === 0) return 1;
    const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
    return Math.sqrt(variance) || 1;
};

const calculateRSI = (closes, period = 14) => {
    if (closes.length <= period) return 50;
    let gains = 0, losses = 0;
    for (let i = closes.length - period; i < closes.length; i++) {
        const change = closes[i] - closes[i - 1];
        if (change > 0) gains += change;
        else losses -= change;
    }
    const avgGain = gains / period;
    const avgLoss = losses / period;
    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
};

async function transmitTelemetry(ticker, score, action, msg) {
    const { error } = await supabase.from('neptune_logs').insert([{
        ticker: ticker, inference_score: score, action: action, mode: PAPER_TRADING ? 'paper' : 'live', log_message: msg.toLowerCase()
    }]);
    if (error) logCyan(`telemetry transmission failed: ${error.message}`);
}

async function manageOpenPositions() {
    try {
        const { data: openPositions, error } = await supabase
            .from('neptune_positions')
            .select('*')
            .eq('status', 'open')
            .eq('mode', PAPER_TRADING ? 'paper' : 'live');

        if (error) throw error;
        if (!openPositions || openPositions.length === 0) return;

        for (const pos of openPositions) {
            // Fetch latest candle strictly for the open position's ticker
            const latestCandles = await exchange.fetchOHLCV(pos.ticker, TIMEFRAME, undefined, 2);
            if (!latestCandles || latestCandles.length < 2) continue;
            
            const currentCandle = latestCandles[latestCandles.length - 2];
            const currentHigh = currentCandle[2];
            const currentLow = currentCandle[3];
            const currentClose = currentCandle[4];

            const nextCandleCount = pos.candles_held + 1;
            let activeStopFloor = parseFloat(pos.stop_loss);
            let exitReason = null;
            let finalPrice = currentClose;

            // 🛡️ DYNAMIC BREAKEVEN SHIELD
            if (nextCandleCount >= 5) activeStopFloor = parseFloat(pos.entry_price);

            if (currentLow <= activeStopFloor) {
                exitReason = nextCandleCount >= 5 ? 'breakeven' : 'loss';
                finalPrice = activeStopFloor;
            } else if (currentHigh >= parseFloat(pos.take_profit)) {
                exitReason = 'win';
                finalPrice = parseFloat(pos.take_profit);
            } else if (nextCandleCount >= riskSettings.forwardHorizonCandles) {
                exitReason = 'timeout';
                finalPrice = currentClose;
            }

            if (exitReason) {
                const profitPct = ((finalPrice - parseFloat(pos.entry_price)) / parseFloat(pos.entry_price)) * 100;
                await supabase.from('neptune_positions').update({ status: exitReason, candles_held: nextCandleCount }).eq('id', pos.id);
                const logMsg = `position closed via ${exitReason} at ${finalPrice}. return: ${profitPct.toFixed(2)}%`;
                logCyan(`>>> [${pos.ticker}] ${logMsg}`);
                await transmitTelemetry(pos.ticker, 0, exitReason, logMsg);
            } else {
                await supabase.from('neptune_positions').update({ candles_held: nextCandleCount }).eq('id', pos.id);
            }
        }
    } catch (err) {
        logCyan(`error running portfolio state manager: ${err.message}`);
    }
}

async function extractFeaturesAndScore(botConfig, session) {
    const candles = await exchange.fetchOHLCV(botConfig.ticker, TIMEFRAME, undefined, 300);
    const minHistoryRequired = Math.max(settings.macro_ema_slow, settings.htf_filter_period) + 60;
    
    if (!candles || candles.length < minHistoryRequired) return null;

    const idx = candles.length - 2;
    const timestamp = candles[idx][0];
    const opens = candles.map(curr => parseFloat(curr[1]));
    const highs = candles.map(curr => parseFloat(curr[2]));
    const lows = candles.map(curr => parseFloat(curr[3]));
    const closes = candles.map(curr => parseFloat(curr[4]));
    const volumes = candles.map(curr => parseFloat(curr[5]));

    const currentClose = closes[idx];
    const currentOpen = opens[idx];
    const currentHigh = highs[idx];
    const currentLow = lows[idx];

    const date = new Date(timestamp);
    const hour = date.getUTCHours();
    const isLondonOpen = (hour >= 7 && hour <= 10) ? 1.0 : 0.0;
    const isNewYorkOpen = (hour >= 12 && hour <= 15) ? 1.0 : 0.0;
    const isAsianSqueeze = (hour >= 0 && hour <= 4) ? 1.0 : 0.0;

    if (isAsianSqueeze === 1.0) return null;

    const hour_sin = Math.sin(2 * Math.PI * hour / 24.0);
    const hour_cos = Math.cos(2 * Math.PI * hour / 24.0);

    let trSum14 = 0;
    for (let j = idx - 13; j <= idx; j++) {
        trSum14 += Math.max(highs[j] - lows[j], Math.abs(highs[j] - (closes[j-1] || opens[j])), Math.abs(lows[j] - (closes[j-1] || opens[j])));
    }
    const rawATR = trSum14 / 14; 
    
    let trSum7 = 0;
    for (let j = idx - 6; j <= idx; j++) {
        trSum7 += Math.max(highs[j] - lows[j], Math.abs(highs[j] - (closes[j-1] || opens[j])), Math.abs(lows[j] - (closes[j-1] || opens[j])));
    }
    const shortATR = trSum7 / 7;
    const atr_squeeze_ratio = shortATR / rawATR; 
    const atr_percentage = (rawATR / currentClose) * 100;

    const totalFriction = riskSettings.takerFeePerc + riskSettings.priceImpactPerc;
    if ((rawATR / currentClose) < (totalFriction * riskSettings.minVolFeeMultiplier)) return null;

    const volSlice = volumes.slice(idx - 59, idx + 1);
    const volMean = volSlice.reduce((a, b) => a + b, 0) / 60;
    const vol_z_score = (volumes[idx] - volMean) / calcStdDev(volSlice, volMean); 
    const rvol = volumes[idx] / (volumes.slice(idx - 19, idx + 1).reduce((a, b) => a + b, 0) / 20 || 1);
    const candleRange = Math.max(currentHigh - currentLow, 0.00001);
    const vol_delta_ratio = (((currentClose - currentOpen) / candleRange) * volumes[idx]) / (volMean || 1);

    const body_to_range_ratio = Math.max(Math.abs(currentClose - currentOpen), 0.0001) / candleRange;
    const upper_wick_ratio = (currentHigh - Math.max(currentOpen, currentClose)) / candleRange;
    const lower_wick_ratio = (Math.min(currentOpen, currentClose) - currentLow) / candleRange;

    const ranges20 = [];
    for (let j = idx - 19; j <= idx; j++) ranges20.push(highs[j] - lows[j]);
    const rangeMean20 = ranges20.reduce((a, b) => a + b, 0) / 20;
    const volatility_z_score = (candleRange - rangeMean20) / calcStdDev(ranges20, rangeMean20);

    const closeSliceRSI = closes.slice(0, idx + 1);
    const rsi_14 = calculateRSI(closeSliceRSI, 14);
    const rsi_slope = (rsi_14 - calculateRSI(closeSliceRSI.slice(0, closeSliceRSI.length - 3), 14)) / 3;

    const emaFast = calculateEMAArray(closeSliceRSI, settings.ema_fast_period).pop();
    const emaSlow = calculateEMAArray(closeSliceRSI, settings.ema_slow_period).pop();
    const ema_spread_ratio = (emaFast - emaSlow) / emaSlow;

    let bodySum5 = 0, rangeSum5 = 0;
    for (let j = idx - 4; j <= idx; j++) {
        bodySum5 += Math.abs(closes[j] - opens[j]);
        rangeSum5 += Math.max(highs[j] - lows[j], 0.00001);
    }
    const price_density = bodySum5 / rangeSum5;
    const momentum_3_period = (currentClose - closes[idx - 3]) / (closes[idx - 3] || 1);
    const momentum_12_period = (currentClose - closes[idx - 12]) / (closes[idx - 12] || 1);

    const closeSlice20 = closes.slice(idx - 19, idx + 1);
    const meanClose20 = closeSlice20.reduce((a, b) => a + b, 0) / 20;
    const bb_position = (currentClose - meanClose20) / calcStdDev(closeSlice20, meanClose20);

    let dirPersist = 0;
    for (let j = idx - 4; j <= idx; j++) dirPersist += (closes[j] - opens[j]) / Math.max(highs[j] - lows[j], 0.00001);
    const directional_persistence = dirPersist / 5;

    const macroEmaFast = calculateEMAArray(closeSliceRSI, settings.macro_ema_fast).pop();
    const macroEmaSlow = calculateEMAArray(closeSliceRSI, settings.macro_ema_slow).pop();
    const dist_from_macro_fast = (currentClose - macroEmaFast) / macroEmaFast;
    
    if (dist_from_macro_fast > 0.025) return null; // Macro extension veto

    const dist_from_macro_slow = (currentClose - macroEmaSlow) / macroEmaSlow;
    let market_regime = (currentClose > macroEmaFast && macroEmaFast > macroEmaSlow) ? 1 : ((currentClose < macroEmaFast && macroEmaFast < macroEmaSlow) ? -1 : 0);

    const ema12Array = calculateEMAArray(closeSliceRSI, 12);
    const ema26Array = calculateEMAArray(closeSliceRSI, 26);
    const macdLine = [];
    for (let j = 0; j < closeSliceRSI.length; j++) {
        macdLine.push((j >= closeSliceRSI.length - ema12Array.length ? ema12Array[j - (closeSliceRSI.length - ema12Array.length)] : closeSliceRSI[j]) - (j >= closeSliceRSI.length - ema26Array.length ? ema26Array[j - (closeSliceRSI.length - ema26Array.length)] : closeSliceRSI[j]));
    }
    const macd_hist = (macdLine[macdLine.length - 1] - calculateEMAArray(macdLine, 9).pop()) / (currentClose || 1);

    let colorFlips = 0;
    for (let j = idx; j >= idx - 3; j--) if ((closes[j] >= opens[j]) !== (closes[j-1] >= opens[j-1])) colorFlips++;
    const is_whipsaw = colorFlips >= 3 ? 1 : 0;

    const vptHistory = [];
    for (let j = idx - 19; j <= idx; j++) vptHistory.push(((closes[j] - (closes[j-1] || opens[j])) / (closes[j-1] || opens[j] || 1)) * volumes[j]);
    const vptMean = vptHistory.reduce((a, b) => a + b, 0) / 20;
    const vpt_z_score = (vptHistory[vptHistory.length - 1] - vptMean) / calcStdDev(vptHistory, vptMean);

    const inputArray = new Float32Array([
        isLondonOpen, isNewYorkOpen, isAsianSqueeze, hour_sin, hour_cos, vol_z_score, rvol, vol_delta_ratio, body_to_range_ratio, upper_wick_ratio, lower_wick_ratio, atr_squeeze_ratio, atr_percentage, volatility_z_score, rsi_14, rsi_slope, macd_hist, ema_spread_ratio, price_density, momentum_3_period, momentum_12_period, bb_position, directional_persistence, dist_from_macro_fast, dist_from_macro_slow, market_regime, is_whipsaw, vpt_z_score
    ]);

    const tensorInput = new ort.Tensor('float32', inputArray, [1, 28]);
    const outputMap = await session.run({ float_input: tensorInput });
    const probabilities = outputMap[session.outputNames[1]].data;
    const rawModelScore = probabilities[1] !== undefined ? probabilities[1] : probabilities[0];

    return { ticker: botConfig.ticker, score: rawModelScore, currentClose, rawATR, threshold: botConfig.threshold };
}

async function runExecutionCycle(sessions) {
    try {
        logCyan(`\n[${new Date().toISOString()}] executing multi-pair market scan...`);
        
        // 1. Manage portfolio globally
        await manageOpenPositions();

        // 2. Extract features and infer scores concurrently for the fleet
        const scanPromises = FLEET.map(bot => extractFeaturesAndScore(bot, sessions[bot.ticker]));
        const results = await Promise.all(scanPromises);
        
        let validSignals = [];

        // 3. Filter valid results and build the queue
        for (const res of results) {
            if (!res) continue; // Vetoed by hard shields
            logCyan(`[${res.ticker}] alpha inference score: ${res.score.toFixed(4)}`);
            
            if (res.score >= res.threshold) {
                validSignals.push(res);
            }
        }

        // 4. Resolve FCFS Concurrency (Choose Highest Conviction)
        if (validSignals.length > 0) {
            validSignals.sort((a, b) => b.score - a.score);
            const winner = validSignals[0]; 

            const targetStopLoss = winner.currentClose - (winner.rawATR * riskSettings.atrStopMultiplier);
            const targetTakeProfit = winner.currentClose + (winner.rawATR * riskSettings.atrTargetMultiplier);

            const msg = `highest conviction multi-pair setup detected on ${winner.ticker} (score: ${winner.score.toFixed(4)}). allocating 33% capital slice and initiating bracket orders.`;
            logCyan(`>>> ${msg}`);
            await transmitTelemetry(winner.ticker, winner.score, 'execute', msg);
            
            const { error: posError } = await supabase.from('neptune_positions').insert([{
                ticker: winner.ticker,
                entry_price: winner.currentClose,
                stop_loss: targetStopLoss,
                take_profit: targetTakeProfit,
                status: 'open',
                candles_held: 0,
                raw_atr: winner.rawATR,
                mode: PAPER_TRADING ? 'paper' : 'live'
            }]);

            if (posError) logCyan(`failed to write state instance: ${posError.message}`);

            if (!PAPER_TRADING) {
                // execute ccxt live market orders using 33% sizing calculation here
            }
        } else {
            logCyan(`no inference scores crossed thresholds across fleet. holding flat.`);
        }

    } catch (err) {
        logCyan(`network/api error during cycle: ${err.message}`);
    }
}

async function bootNeptune() {
    logCyan('=================================================');
    logCyan('      project neptune multi-pair initialized');
    logCyan('=================================================');
    
    try {
        const sessions = {};
        for (const bot of FLEET) {
            logCyan(`loading ONNX engine for ${bot.ticker}...`);
            sessions[bot.ticker] = await ort.InferenceSession.create(bot.modelFile);
        }
        
        await runExecutionCycle(sessions);
        
        cron.schedule('0,15,30,45 * * * *', () => {
            runExecutionCycle(sessions);
        });
        
        logCyan('listening for strict 15m multi-pair horizons...');
    } catch (err) {
        logCyan(`❌ fatal initialization error: ${err.message}`);
    }
}

bootNeptune();
