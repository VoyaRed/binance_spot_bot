// live_executor.js - project neptune live execution engine
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
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// configuration
const TICKER = 'ETH/USDT';
const TIMEFRAME = '15m';
const MODEL_FILENAME = 'veto_engine_spot_15m_eth.onnx';
const VETO_THRESHOLD = 0.4807; // Synced with your backtester lock
const PAPER_TRADING = process.env.PAPER_TRADING === 'true';

const exchange = new ccxt.binanceus({
    apiKey: process.env.BINANCEUS_API_KEY,
    secret: process.env.BINANCEUS_API_SECRET,
    enableRateLimit: true
});

const logCyan = (msg) => console.log(`\x1b[36m${msg.toLowerCase()}\x1b[0m`);

// --- SYNCHRONIZED MATH CLOSURES ---
const settings = {
    ema_fast_period: 9,     
    ema_slow_period: 21,    
    macro_ema_fast: 50,     
    macro_ema_slow: 200,
    htf_filter_period: 160  
};

const riskSettings = {
    atrStopMultiplier: 2.0,
    atrTargetMultiplier: 2.0,
    forwardHorizonCandles: 12,
    // --- friction parameters for volatility filter ---
    priceImpactPerc: 0.0002,    
    takerFeePerc: 0.00019,      
    minVolFeeMultiplier: 6.0    
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

async function transmitTelemetry(score, action, msg) {
    const { error } = await supabase.from('neptune_logs').insert([{
        ticker: TICKER,
        inference_score: score,
        action: action,
        mode: PAPER_TRADING ? 'paper' : 'live',
        log_message: msg.toLowerCase()
    }]);
    if (error) logCyan(`telemetry transmission failed: ${error.message}`);
}

async function manageOpenPositions(currentHigh, currentLow, currentClose) {
    try {
        const { data: openPositions, error } = await supabase
            .from('neptune_positions')
            .select('*')
            .eq('status', 'open')
            .eq('mode', PAPER_TRADING ? 'paper' : 'live');

        if (error) throw error;
        if (!openPositions || openPositions.length === 0) return;

        for (const pos of openPositions) {
            const nextCandleCount = pos.candles_held + 1;
            let activeStopFloor = parseFloat(pos.stop_loss);
            let exitReason = null;
            let finalPrice = currentClose;

            // 🛡️ DYNAMIC BREAKEVEN SHIELD SYNCED FROM BACKTESTER
            if (nextCandleCount >= 5) {
                activeStopFloor = parseFloat(pos.entry_price);
            }

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
                
                await supabase
                    .from('neptune_positions')
                    .update({ status: exitReason, candles_held: nextCandleCount })
                    .eq('id', pos.id);

                const logMsg = `position closed via ${exitReason} at ${finalPrice}. estimated lifecycle return: ${profitPct.toFixed(2)}%`;
                logCyan(`>>> ${logMsg}`);
                await transmitTelemetry(0, exitReason, logMsg);
            } else {
                await supabase
                    .from('neptune_positions')
                    .update({ candles_held: nextCandleCount })
                    .eq('id', pos.id);
            }
        }
    } catch (err) {
        logCyan(`error running portfolio state manager: ${err.message}`);
    }
}

async function runExecutionCycle(session) {
    try {
        logCyan(`\n[${new Date().toISOString()}] fetching latest ${TIMEFRAME} market data for ${TICKER}...`);
        
        const candles = await exchange.fetchOHLCV(TICKER, TIMEFRAME, undefined, 300);
        const minHistoryRequired = Math.max(settings.macro_ema_slow, settings.htf_filter_period) + 60;
        
        if (!candles || candles.length < minHistoryRequired) {
            logCyan(`error: insufficient data fetched. Need ${minHistoryRequired}, got ${candles ? candles.length : 0}.`);
            return;
        }

        // --- SHIFT INDEX TO TARGET LAST FULLY CLOSED CANDLE ---
        const idx = candles.length - 2;
        const currentCandle = candles[idx];
        const timestamp = currentCandle[0];

        const opens = candles.map(curr => parseFloat(curr[1]));
        const highs = candles.map(curr => parseFloat(curr[2]));
        const lows = candles.map(curr => parseFloat(curr[3]));
        const closes = candles.map(curr => parseFloat(curr[4]));
        const volumes = candles.map(curr => parseFloat(curr[5]));

        const currentClose = closes[idx];
        const currentOpen = opens[idx];
        const currentHigh = highs[idx];
        const currentLow = lows[idx];

        // --- MANAGE CURRENT OPEN EXPOSURES BEFORE NEW INFERENCE ---
        await manageOpenPositions(currentHigh, currentLow, currentClose);

        // --- FEATURE EXTRACTION ---
        const date = new Date(timestamp);
        const hour = date.getUTCHours();
        const isLondonOpen = (hour >= 7 && hour <= 10) ? 1.0 : 0.0;
        const isNewYorkOpen = (hour >= 12 && hour <= 15) ? 1.0 : 0.0;
        const isAsianSqueeze = (hour >= 0 && hour <= 4) ? 1.0 : 0.0;

        // 🛡️ SHIELD 1: ASIAN SESSION SQUEEZE DROP
        if (isAsianSqueeze === 1.0) {
            const shieldMsg = "shield active: skipping execution cycle during asian session squeeze.";
            logCyan(shieldMsg);
            await transmitTelemetry(0, 'shield_skip', shieldMsg);
            return;
        }

        const hour_sin = Math.sin(2 * Math.PI * hour / 24.0);
        const hour_cos = Math.cos(2 * Math.PI * hour / 24.0);

        let trSum14 = 0;
        for (let j = idx - 13; j <= idx; j++) {
            const highLow = highs[j] - lows[j];
            const highClose = Math.abs(highs[j] - (closes[j-1] || opens[j]));
            const lowClose = Math.abs(lows[j] - (closes[j-1] || opens[j]));
            trSum14 += Math.max(highLow, highClose, lowClose);
        }
        const rawATR = trSum14 / 14; 
        
        let trSum7 = 0;
        for (let j = idx - 6; j <= idx; j++) {
            const highLow = highs[j] - lows[j];
            const highClose = Math.abs(highs[j] - (closes[j-1] || opens[j]));
            const lowClose = Math.abs(lows[j] - (closes[j-1] || opens[j]));
            trSum7 += Math.max(highLow, highClose, lowClose);
        }
        const shortATR = trSum7 / 7;
        const atr_squeeze_ratio = shortATR / rawATR; 
        const atr_percentage = (rawATR / currentClose) * 100;

        // 🛡️ SHIELD 2: DYNAMIC FEE-VOLATILITY DROP
        const totalFriction = riskSettings.takerFeePerc + riskSettings.priceImpactPerc;
        const decimalATR = rawATR / currentClose;

        if (decimalATR < (totalFriction * riskSettings.minVolFeeMultiplier)) {
            const shieldMsg = `shield active: low volatility fee trap detected. atr decimal (${decimalATR.toFixed(6)}) is below friction threshold.`;
            logCyan(shieldMsg);
            await transmitTelemetry(0, 'shield_skip', shieldMsg);
            return;
        }

        const volSlice = volumes.slice(idx - 59, idx + 1);
        const volMean = volSlice.reduce((a, b) => a + b, 0) / 60;
        const volStdDev = calcStdDev(volSlice, volMean);
        const vol_z_score = (volumes[idx] - volMean) / volStdDev; 

        const volSMA20 = volumes.slice(idx - 19, idx + 1).reduce((a, b) => a + b, 0) / 20;
        const rvol = volumes[idx] / (volSMA20 || 1);

        const candleRange = Math.max(currentHigh - currentLow, 0.00001);
        const signedDelta = ((currentClose - currentOpen) / candleRange) * volumes[idx];
        const vol_delta_ratio = signedDelta / (volMean || 1);

        const bodySize = Math.max(Math.abs(currentClose - currentOpen), 0.0001);
        const body_to_range_ratio = bodySize / candleRange;
        const upperWick = currentHigh - Math.max(currentOpen, currentClose);
        const lowerWick = Math.min(currentOpen, currentClose) - currentLow;
        const upper_wick_ratio = upperWick / candleRange;
        const lower_wick_ratio = lowerWick / candleRange;

        const ranges20 = [];
        for (let j = idx - 19; j <= idx; j++) {
            ranges20.push(highs[j] - lows[j]);
        }
        const rangeMean20 = ranges20.reduce((a, b) => a + b, 0) / 20;
        const rangeStd20 = calcStdDev(ranges20, rangeMean20);
        const volatility_z_score = (candleRange - rangeMean20) / rangeStd20;

        const closeSliceRSI = closes.slice(0, idx + 1);
        const rsi_14 = calculateRSI(closeSliceRSI, 14);
        const pastRsi = calculateRSI(closeSliceRSI.slice(0, closeSliceRSI.length - 3), 14);
        const rsi_slope = (rsi_14 - pastRsi) / 3;

        const emaFast = calculateEMAArray(closeSliceRSI, settings.ema_fast_period).pop();
        const emaSlow = calculateEMAArray(closeSliceRSI, settings.ema_slow_period).pop();
        const ema_spread_ratio = (emaFast - emaSlow) / emaSlow;

        let bodySum5 = 0;
        let rangeSum5 = 0;
        for (let j = idx - 4; j <= idx; j++) {
            bodySum5 += Math.abs(closes[j] - opens[j]);
            rangeSum5 += Math.max(highs[j] - lows[j], 0.00001);
        }
        const price_density = bodySum5 / rangeSum5;

        const momentum_3_period = (currentClose - closes[idx - 3]) / (closes[idx - 3] || 1);
        const momentum_12_period = (currentClose - closes[idx - 12]) / (closes[idx - 12] || 1);

        const closeSlice20 = closes.slice(idx - 19, idx + 1);
        const meanClose20 = closeSlice20.reduce((a, b) => a + b, 0) / 20;
        const stdDevClose20 = calcStdDev(closeSlice20, meanClose20);
        const bb_position = (currentClose - meanClose20) / stdDevClose20;

        let dirPersist = 0;
        for (let j = idx - 4; j <= idx; j++) {
            const r = Math.max(highs[j] - lows[j], 0.00001);
            dirPersist += (closes[j] - opens[j]) / r;
        }
        const directional_persistence = dirPersist / 5;

        const macroEmaFast = calculateEMAArray(closeSliceRSI, settings.macro_ema_fast).pop();
        const macroEmaSlow = calculateEMAArray(closeSliceRSI, settings.macro_ema_slow).pop();
        const dist_from_macro_fast = (currentClose - macroEmaFast) / macroEmaFast;
        const dist_from_macro_slow = (currentClose - macroEmaSlow) / macroEmaSlow;

        // 🛡️ SHIELD 3: MACRO EXTENSION VETO (HARD CEILING)
        if (dist_from_macro_fast > 0.025) {
            const shieldMsg = `shield active: overextended momentum spike detected. distance from macro fast (${dist_from_macro_fast.toFixed(4)}) exceeds 2.5% ceiling.`;
            logCyan(shieldMsg);
            await transmitTelemetry(0, 'shield_skip', shieldMsg);
            return;
        }

        let market_regime = 0; 
        if (currentClose > macroEmaFast && macroEmaFast > macroEmaSlow) market_regime = 1; 
        else if (currentClose < macroEmaFast && macroEmaFast < macroEmaSlow) market_regime = -1;

        const ema12Array = calculateEMAArray(closeSliceRSI, 12);
        const ema26Array = calculateEMAArray(closeSliceRSI, 26);
        const macdLine = [];
        const startIdx12 = closeSliceRSI.length - ema12Array.length;
        const startIdx26 = closeSliceRSI.length - ema26Array.length;
        for (let j = 0; j < closeSliceRSI.length; j++) {
            const val12 = j >= startIdx12 ? ema12Array[j - startIdx12] : closeSliceRSI[j];
            const val26 = j >= startIdx26 ? ema26Array[j - startIdx26] : closeSliceRSI[j];
            macdLine.push(val12 - val26);
        }
        const signalLine = calculateEMAArray(macdLine, 9);
        const currentMacd = macdLine[macdLine.length - 1];
        const currentSignal = signalLine[signalLine.length - 1] || 0;
        const currentHist = currentMacd - currentSignal;
        const macd_hist = currentHist / (currentClose || 1);

        let colorFlips = 0;
        for (let j = idx; j >= idx - 3; j--) {
            const currentColor = closes[j] >= opens[j] ? 'green' : 'red';
            const prevColor = closes[j-1] >= opens[j-1] ? 'green' : 'red';
            if (currentColor !== prevColor) colorFlips++;
        }
        const is_whipsaw = colorFlips >= 3 ? 1 : 0;

        const vptHistory = [];
        for (let j = idx - 19; j <= idx; j++) {
            const prevC = closes[j-1] || opens[j];
            const pctChange = (closes[j] - prevC) / (prevC || 1);
            vptHistory.push(pctChange * volumes[j]);
        }
        const vptMean = vptHistory.reduce((a, b) => a + b, 0) / 20;
        const vptStd = calcStdDev(vptHistory, vptMean);
        const vpt_z_score = (vptHistory[vptHistory.length - 1] - vptMean) / vptStd;

        // --- ONNX INFERENCE ---
        const inputArray = new Float32Array([
            isLondonOpen, isNewYorkOpen, isAsianSqueeze,
            hour_sin, hour_cos, vol_z_score, rvol, vol_delta_ratio,
            body_to_range_ratio, upper_wick_ratio, lower_wick_ratio,
            atr_squeeze_ratio, atr_percentage, volatility_z_score,
            rsi_14, rsi_slope, macd_hist, ema_spread_ratio,
            price_density, momentum_3_period, momentum_12_period,
            bb_position, directional_persistence, dist_from_macro_fast,
            dist_from_macro_slow, market_regime, is_whipsaw, vpt_z_score
        ]);

        const tensorInput = new ort.Tensor('float32', inputArray, [1, 28]);
        const outputMap = await session.run({ float_input: tensorInput });
        
        const probKey = session.outputNames[1];
        const probabilities = outputMap[probKey].data;
        const rawModelScore = probabilities[1] !== undefined ? probabilities[1] : probabilities[0];

        logCyan(`alpha inference score: ${rawModelScore.toFixed(4)}`);

        if (rawModelScore >= VETO_THRESHOLD) {
            const targetStopLoss = currentClose - (rawATR * riskSettings.atrStopMultiplier);
            const targetTakeProfit = currentClose + (rawATR * riskSettings.atrTargetMultiplier);

            const msg = `high conviction setup detected. initiating bracket orders at ${currentClose}.`;
            logCyan(`>>> ${msg}`);
            await transmitTelemetry(rawModelScore, 'execute', msg);
            
            const { error: posError } = await supabase.from('neptune_positions').insert([{
                ticker: TICKER,
                entry_price: currentClose,
                stop_loss: targetStopLoss,
                take_profit: targetTakeProfit,
                status: 'open',
                candles_held: 0,
                raw_atr: rawATR,
                mode: PAPER_TRADING ? 'paper' : 'live'
            }]);

            if (posError) logCyan(`failed to write state instance to database: ${posError.message}`);

            if (!PAPER_TRADING) {
                // execute ccxt live market orders here
            }
        } else {
            const msg = `inference score below threshold. holding flat.`;
            logCyan(msg);
            await transmitTelemetry(rawModelScore, 'hold', msg);
        }

    } catch (err) {
        logCyan(`network/api error during cycle: ${err.message}`);
        await transmitTelemetry(0, 'error', err.message);
    }
}

async function bootNeptune() {
    logCyan('=================================================');
    logCyan('         project neptune initialized');
    logCyan('=================================================');
    
    try {
        const session = await ort.InferenceSession.create(MODEL_FILENAME);
        
        await runExecutionCycle(session);
        
        // --- PRECISION CRON SCHEDULER: SNAPS TO 00, 15, 30, 45 ---
        cron.schedule('0,15,30,45 * * * *', () => {
            runExecutionCycle(session);
        });
        
        logCyan('listening for strict 15m market horizons...');
    } catch (err) {
        logCyan(`❌ fatal initialization error: ${err.message}`);
    }
}

bootNeptune();
                                                         
