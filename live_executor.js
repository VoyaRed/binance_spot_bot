// live_executor.js - project neptune live execution engine
require('dotenv').config();
const ccxt = require('ccxt');
const ort = require('onnxruntime-node');

// configuration
const TICKER = 'ETH/USDT';
const TIMEFRAME = '15m';
const MODEL_FILENAME = 'veto_engine_spot_15m_eth.onnx';

// locked p90 threshold from xgboost walk-forward validation
const VETO_THRESHOLD = 0.90; 
const PAPER_TRADING = process.env.PAPER_TRADING === 'true';

const exchange = new ccxt.binanceus({
    apiKey: process.env.BINANCEUS_API_KEY,
    secret: process.env.BINANCEUS_API_SECRET,
    enableRateLimit: true
});

// synchronized multi-timeframe mapping structures
const settings = {
    ema_fast_period: 9,     
    ema_slow_period: 21,    
    macro_ema_fast: 50,     
    macro_ema_slow: 200,
    htf_filter_period: 160  
};

const tradeSettings = {
    riskPerTradePercent: 0.025, 
    maxCapitalExposure: 0.98,   
};

const riskSettings = {
    atrStopMultiplier: 2.0,     
    atrTargetMultiplier: 2.0,   
    priceImpactPerc: 0.0002,    
    takerFeePerc: 0.00019,      
    minVolFeeMultiplier: 6.0          
};

// --- synchronized math closures ---
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

const logCyan = (msg) => console.log(`\x1b[36m${msg.toLowerCase()}\x1b[0m`);

async function runExecutionCycle(session) {
    try {
        logCyan(`\n[${new Date().toISOString()}] fetching latest ${TIMEFRAME} market data for ${TICKER}...`);
        
        // fetch enough history to calculate the 200 macro ema
        const candles = await exchange.fetchOHLCV(TICKER, TIMEFRAME, undefined, 300);
        if (!candles || candles.length < 250) {
            logCyan('error: insufficient data fetched from binance.us.');
            return;
        }

        const idx = candles.length - 1;
        const currentCandle = candles[idx];
        const timestamp = currentCandle[0];
        
        const opens = candles.map(c => parseFloat(c[1]));
        const highs = candles.map(c => parseFloat(c[2]));
        const lows = candles.map(c => parseFloat(c[3]));
        const closes = candles.map(c => parseFloat(c[4]));
        const volumes = candles.map(c => parseFloat(c[5]));

        const currentClose = closes[idx];
        const currentOpen = opens[idx];
        const currentHigh = highs[idx];
        const currentLow = lows[idx];

        // --- session clocks ---
        const date = new Date(timestamp);
        const hour = date.getUTCHours();
        const isLondonOpen = (hour >= 7 && hour <= 10) ? 1.0 : 0.0;
        const isNewYorkOpen = (hour >= 12 && hour <= 15) ? 1.0 : 0.0;
        const isAsianSqueeze = (hour >= 0 && hour <= 4) ? 1.0 : 0.0;

        if (isAsianSqueeze === 1.0) {
            logCyan('asian session active. model vetoes execution.');
            return;
        }

        const hour_sin = Math.sin(2 * Math.PI * hour / 24.0);
        const hour_cos = Math.cos(2 * Math.PI * hour / 24.0);

        // --- volatility & atr matrices ---
        let trSum14 = 0;
        for (let j = closes.length - 14; j < closes.length; j++) {
            const highLow = highs[j] - lows[j];
            const highClose = Math.abs(highs[j] - closes[j-1]);
            const lowClose = Math.abs(lows[j] - closes[j-1]);
            trSum14 += Math.max(highLow, highClose, lowClose);
        }
        const rawATR = trSum14 / 14; 
        
        let trSum7 = 0;
        for (let j = closes.length - 7; j < closes.length; j++) {
            const highLow = highs[j] - lows[j];
            const highClose = Math.abs(highs[j] - closes[j-1]);
            const lowClose = Math.abs(lows[j] - closes[j-1]);
            trSum7 += Math.max(highLow, highClose, lowClose);
        }
        const shortATR = trSum7 / 7;
        const atr_squeeze_ratio = shortATR / rawATR; 
        const atr_percentage = (rawATR / currentClose) * 100;

        const totalFriction = riskSettings.takerFeePerc + riskSettings.priceImpactPerc;
        const decimalATR = rawATR / currentClose;
        
        if (decimalATR < (totalFriction * riskSettings.minVolFeeMultiplier)) {
            logCyan('fee-volatility shield active. spread too tight to execute.');
            return;
        }

        // --- volume profile dynamics ---
        const volSlice = volumes.slice(-60);
        const volMean = volSlice.reduce((a, b) => a + b, 0) / 60;
        const volStdDev = calcStdDev(volSlice, volMean);
        const vol_z_score = (volumes[idx] - volMean) / volStdDev; 

        const volSMA20 = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;
        const rvol = volumes[idx] / (volSMA20 || 1);

        const candleRange = Math.max(currentHigh - currentLow, 0.00001);
        const signedDelta = ((currentClose - currentOpen) / candleRange) * volumes[idx];
        const vol_delta_ratio = signedDelta / (volMean || 1);

        // --- candle geometry ---
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

        // --- oscillators & momentum ---
        const rsi_14 = calculateRSI(closes, 14);
        const pastRsi = calculateRSI(closes.slice(0, closes.length - 3), 14);
        const rsi_slope = (rsi_14 - pastRsi) / 3;

        const emaFast = calculateEMAArray(closes, settings.ema_fast_period).pop();
        const emaSlow = calculateEMAArray(closes, settings.ema_slow_period).pop();
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

        const closeSlice20 = closes.slice(-20);
        const meanClose20 = closeSlice20.reduce((a, b) => a + b, 0) / 20;
        const stdDevClose20 = calcStdDev(closeSlice20, meanClose20);
        const bb_position = (currentClose - meanClose20) / stdDevClose20;

        let dirPersist = 0;
        for (let j = idx - 4; j <= idx; j++) {
            const r = Math.max(highs[j] - lows[j], 0.00001);
            dirPersist += (closes[j] - opens[j]) / r;
        }
        const directional_persistence = dirPersist / 5;

        // --- macro timeframe projections ---
        const macroEmaFast = calculateEMAArray(closes, settings.macro_ema_fast).pop();
        const macroEmaSlow = calculateEMAArray(closes, settings.macro_ema_slow).pop();
        const dist_from_macro_fast = (currentClose - macroEmaFast) / macroEmaFast;
        const dist_from_macro_slow = (currentClose - macroEmaSlow) / macroEmaSlow;

        if (dist_from_macro_fast > 0.025) {
            logCyan('macro extension veto triggered. flat.');
            return;
        }

        let market_regime = 0; 
        if (currentClose > macroEmaFast && macroEmaFast > macroEmaSlow) market_regime = 1; 
        else if (currentClose < macroEmaFast && macroEmaFast < macroEmaSlow) market_regime = -1;

        const ema12Array = calculateEMAArray(closes, 12);
        const ema26Array = calculateEMAArray(closes, 26);
        const macdLine = [];
        const startIdx12 = closes.length - ema12Array.length;
        const startIdx26 = closes.length - ema26Array.length;
        for (let j = 0; j < closes.length; j++) {
            const val12 = j >= startIdx12 ? ema12Array[j - startIdx12] : closes[j];
            const val26 = j >= startIdx26 ? ema26Array[j - startIdx26] : closes[j];
            macdLine.push(val12 - val26);
        }
        const signalLine = calculateEMAArray(macdLine, 9);
        const currentHist = macdLine[macdLine.length - 1] - (signalLine[signalLine.length - 1] || 0);
        const macd_hist = currentHist / (currentClose || 1);

        let colorFlips = 0;
        for (let j = closes.length - 1; j >= closes.length - 4; j--) {
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

        // --- onnx neural net execution ---
        const inputArray = new Float32Array([
            isLondonOpen, isNewYorkOpen, isAsianSqueeze, hour_sin, hour_cos, 
            vol_z_score, rvol, vol_delta_ratio, body_to_range_ratio, 
            upper_wick_ratio, lower_wick_ratio, atr_squeeze_ratio, 
            atr_percentage, volatility_z_score, rsi_14, rsi_slope, 
            macd_hist, ema_spread_ratio, price_density, momentum_3_period, 
            momentum_12_period, bb_position, directional_persistence, 
            dist_from_macro_fast, dist_from_macro_slow, market_regime, 
            is_whipsaw, vpt_z_score
        ]);

        const tensorInput = new ort.Tensor('float32', inputArray, [1, 28]);
        const outputMap = await session.run({ float_input: tensorInput });
        
        const probKey = session.outputNames[1];
        const probabilities = outputMap[probKey].data;
        const rawModelScore = probabilities[1] !== undefined ? probabilities[1] : probabilities[0];

        logCyan(`alpha inference score: ${rawModelScore.toFixed(4)}`);

        if (rawModelScore >= VETO_THRESHOLD) {
            logCyan(`\n>>> high conviction setup detected! (score: ${rawModelScore.toFixed(4)} >= threshold: ${VETO_THRESHOLD})`);
            
            // --- dynamic position sizing ---
            const balance = await exchange.fetchBalance();
            const availableUSDT = balance.free['USDT'];
            
            const stopLoss = currentClose - (rawATR * riskSettings.atrStopMultiplier);
            const distanceToStop = currentClose - stopLoss;
            const cashRisk = availableUSDT * tradeSettings.riskPerTradePercent;
            let unitsToBuy = cashRisk / distanceToStop;
            
            const maxSpotUnits = (availableUSDT * tradeSettings.maxCapitalExposure) / currentClose;
            if (unitsToBuy > maxSpotUnits) unitsToBuy = maxSpotUnits;

            const takeProfitPrice = currentClose + (rawATR * riskSettings.atrTargetMultiplier);

            logCyan(`calculated position size: ${unitsToBuy.toFixed(4)} ${TICKER}`);
            logCyan(`stop loss limit: ${stopLoss.toFixed(2)} | take profit: ${takeProfitPrice.toFixed(2)}`);

            if (PAPER_TRADING) {
                logCyan('📝 paper trading mode active. execution simulated successfully. awaiting next horizon.');
            } else {
                logCyan('⚠️ live execution mode active. sending orders to binance.us...');
                const marketOrder = await exchange.createMarketBuyOrder(TICKER, unitsToBuy);
                logCyan(`✅ market order filled: ${marketOrder.id}`);

                // generate oco (one-cancels-the-other) bracket order for tp/sl
                // requires specific ccxt binance implementation parameters
                await exchange.createOrder(TICKER, 'stop_loss_limit', 'sell', unitsToBuy, stopLoss, {
                    'stopPrice': stopLoss,
                    'takeProfitPrice': takeProfitPrice
                });
                logCyan(`🛡️ bracket limits set.`);
            }
        } else {
            logCyan(`inference score below threshold. holding flat.`);
        }

    } catch (err) {
        logCyan(`network/api error during cycle: ${err.message}`);
    }
}

async function bootNeptune() {
    logCyan('=================================================');
    logCyan('         project neptune initialized');
    logCyan('=================================================');
    
    if (PAPER_TRADING) {
        logCyan('mode: paper trading (live market data, simulated orders)');
    } else {
        logCyan('mode: live execution (real capital at risk)');
    }

    try {
        const session = await ort.InferenceSession.create(MODEL_FILENAME);
        logCyan(`✅ onnx machine learning model successfully loaded.`);
        
        // Run immediately on startup, then interval
        await runExecutionCycle(session);
        
        // 15-minute interval clock (900,000 milliseconds)
        setInterval(() => runExecutionCycle(session), 15 * 60 * 1000);
        
        logCyan('listening for market horizons...');
    } catch (err) {
        logCyan(`❌ fatal initialization error: ${err.message}`);
    }
}

bootNeptune();
