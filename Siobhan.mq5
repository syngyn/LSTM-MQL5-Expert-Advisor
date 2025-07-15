//+------------------------------------------------------------------+
//|                                                        Siobhan.mq5|
//|                                         Copyright 2025, Jason Rusk|
//|  Dedicated to my late wife who always supported my trading passion|
//|                                                                   |
//+------------------------------------------------------------------+

#property copyright "Jason.W.Rusk@gmail.com 2025"
#property version   "1.42" // FIX: Use FILE_COMMON for backtest predictions

#include <Trade/Trade.mqh>
#include <Files/File.mqh>
#include <stdlib.mqh>
#include <Math/Stat/Math.mqh>

// --- SELF-LEARNING ---
input bool EnableSelfLearning = true; // Enables loading best parameters in live trading.
#define LEARNED_PARAMS_FILE "Siobhan_learned_parameters.csv" // File to store the best optimization parameters

// --- Main Operation Mode ---
enum ENUM_TRADING_MODE { MODE_TRADING_DISABLED, MODE_CLASSIFICATION_ONLY, MODE_REGRESSION_ONLY, MODE_COMBINED };
input ENUM_TRADING_MODE TradingLogicMode = MODE_REGRESSION_ONLY; // Trading modes
input bool              EnablePricePredictionDisplay = true; // Displays price prediction on the chart.

// --- Independent Risk Management Modes ---
enum ENUM_STOP_LOSS_MODE { SL_ATR_BASED, SL_STATIC_PIPS };
enum ENUM_TAKE_PROFIT_MODE { TP_ATR_MULTIPLE, TP_STATIC_PIPS, TP_REGRESSION_TARGET };
input ENUM_STOP_LOSS_MODE   StopLossMode = SL_ATR_BASED;
input ENUM_TAKE_PROFIT_MODE TakeProfitMode = TP_ATR_MULTIPLE;

// --- Selectable Target Bar and TP Execution ---
enum ENUM_TARGET_BAR { H_PLUS_1=0, H_PLUS_2=1, H_PLUS_3=2, H_PLUS_4=3, H_PLUS_5=4 };
input ENUM_TARGET_BAR TakeProfitTargetBar = H_PLUS_5;
input bool UseMarketOrderForTP = false; // Use a market order to exit a position instead of the tp

// --- Trading & Risk Management Inputs (These are defaults and for the optimizer) ---
input double RiskPercent = 3.0;
input double SignalThresholdProbability = 0.70;
input double MinimumRiskRewardRatio = 1.5;
input int    RequiredConsistentSteps = 4;
input int    StaticStopLossPips = 300;
input int    StaticTakeProfitPips = 400;
input int    ATR_Period = 14;
input double ATR_SL_Multiplier = 1.5;

// --- Minimum Profit Filter ---
input double MinProfitPips = 10.0;
// --- Time-Based Exit Settings ---
input bool   EnableTimeBasedExit   = false;
input int    MaxPositionHoldBars   = 8;
input int    InpExitBarMinute      = 58;

// --- Accuracy & Backtesting Inputs ---
input int    AccuracyLookaheadBars = 24;
input string Symbol_EURJPY = "EURJPY", Symbol_USDJPY = "USDJPY", Symbol_GBPUSD = "GBPUSD";
input string Symbol_EURGBP = "EURGBP", Symbol_USDCAD = "USDCAD", Symbol_USDCHF = "USDCHF";

// --- Filter & Trailing Stop Inputs ---
input bool   EnableTrailingStop = true, EnableADXFilter = true;
input double TrailingStartPips = 12.0, TrailingStopPips = 3.0;
input int    ADX_Period = 14, ADX_Threshold = 25;

// --- Model Communication ---
input int RequestTimeout = 5000;
#define PREDICTION_STEPS 5
#define SEQ_LEN 20
#define FEATURE_COUNT 12
#define DATA_FOLDER "LSTM_Trading\\data"

// --- Global Handles & Variables ---
int atr_handle, macd_handle, rsi_handle, stoch_handle, cci_handle, adx_handle;
CTrade trade;
enum ENUM_PREDICTION_DIRECTION { DIR_BULLISH, DIR_BEARISH };
struct PendingPrediction { double target_price; datetime start_time, end_time; ENUM_PREDICTION_DIRECTION direction; int step; };
PendingPrediction g_pending_predictions[];
double g_last_predictions[PREDICTION_STEPS];
double g_accuracy_pct[PREDICTION_STEPS];
int    g_total_hits[PREDICTION_STEPS], g_total_predictions[PREDICTION_STEPS];
double g_active_trade_target_price = 0;

// --- BACKTESTING: Struct and array for pre-computed predictions ---
#define BACKTEST_PREDICTIONS_FILE "backtest_predictions.csv"
struct BacktestPrediction
  {
   datetime timestamp;
   double   buy_prob;
   double   sell_prob;
   double   hold_prob;
   double   predicted_prices[PREDICTION_STEPS];
  };
BacktestPrediction g_backtest_predictions[];
int g_backtest_prediction_idx = 0; // Speeds up searching

// --- Global Parameters (Modifiable copies of the inputs for self-learning) ---
double g_RiskPercent;
double g_SignalThresholdProbability;
double g_MinimumRiskRewardRatio;
int    g_RequiredConsistentSteps;
int    g_StaticStopLossPips;
int    g_StaticTakeProfitPips;
int    g_ATR_Period;
double g_ATR_SL_Multiplier;
double g_MinProfitPips;
bool   g_EnableTimeBasedExit;
int    g_MaxPositionHoldBars;
int    g_ExitBarMinute;
bool   g_EnableTrailingStop;
double g_TrailingStartPips;
double g_TrailingStopPips;
bool   g_EnableADXFilter;
int    g_ADX_Period;
int    g_ADX_Threshold;

//+------------------------------------------------------------------+
//| --- HELPER FUNCTIONS ---
//+------------------------------------------------------------------+

void LoadLearnedParameters()
  {
   if(!EnableSelfLearning) return;
   if(MQLInfoInteger(MQL_TESTER)) return;

   if(!FileIsExist(LEARNED_PARAMS_FILE))
     {
      Print("No learned parameters file found. Using default EA settings.");
      return;
     }
   int file_handle = FileOpen(LEARNED_PARAMS_FILE, FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_WRITE, ',');
   if(file_handle == INVALID_HANDLE)
     {
      PrintFormat("Error opening learned parameters file for reading: %s. Error code: %d", LEARNED_PARAMS_FILE, GetLastError());
      return;
     }
   Print("Found learned parameters file. Overriding default EA settings...");
   while(!FileIsEnding(file_handle))
     {
      string key = FileReadString(file_handle);
      if(key == "RiskPercent") g_RiskPercent = FileReadDouble(file_handle);
      else if(key == "SignalThresholdProbability") g_SignalThresholdProbability = FileReadDouble(file_handle);
      else if(key == "MinimumRiskRewardRatio") g_MinimumRiskRewardRatio = FileReadDouble(file_handle);
      else if(key == "RequiredConsistentSteps") g_RequiredConsistentSteps = (int)FileReadNumber(file_handle);
      else if(key == "StaticStopLossPips") g_StaticStopLossPips = (int)FileReadNumber(file_handle);
      else if(key == "StaticTakeProfitPips") g_StaticTakeProfitPips = (int)FileReadNumber(file_handle);
      else if(key == "ATR_Period") g_ATR_Period = (int)FileReadNumber(file_handle);
      else if(key == "ATR_SL_Multiplier") g_ATR_SL_Multiplier = FileReadDouble(file_handle);
      else if(key == "MinProfitPips") g_MinProfitPips = FileReadDouble(file_handle);
      else if(key == "EnableTimeBasedExit") g_EnableTimeBasedExit = (bool)FileReadNumber(file_handle);
      else if(key == "MaxPositionHoldBars") g_MaxPositionHoldBars = (int)FileReadNumber(file_handle);
      else if(key == "EnableTrailingStop") g_EnableTrailingStop = (bool)FileReadNumber(file_handle);
      else if(key == "TrailingStartPips") g_TrailingStartPips = FileReadDouble(file_handle);
      else if(key == "TrailingStopPips") g_TrailingStopPips = FileReadDouble(file_handle);
      else if(key == "EnableADXFilter") g_EnableADXFilter = (bool)FileReadNumber(file_handle);
      else if(key == "ADX_Period") g_ADX_Period = (int)FileReadNumber(file_handle);
      else if(key == "ADX_Threshold") g_ADX_Threshold = (int)FileReadNumber(file_handle);
      else FileReadString(file_handle);
     }
   FileClose(file_handle);
   Print("Successfully loaded learned parameters.");
   PrintFormat(" -> Loaded RiskPercent: %.2f, Loaded ATR_Period: %d, Loaded ADX_Threshold: %d", g_RiskPercent, g_ATR_Period, g_ADX_Threshold);
  }

void InitializeParameters()
  {
   g_RiskPercent = RiskPercent;
   g_SignalThresholdProbability = SignalThresholdProbability;
   g_MinimumRiskRewardRatio = MinimumRiskRewardRatio;
   g_RequiredConsistentSteps = RequiredConsistentSteps;
   g_StaticStopLossPips = StaticStopLossPips;
   g_StaticTakeProfitPips = StaticTakeProfitPips;
   g_ATR_Period = ATR_Period;
   g_ATR_SL_Multiplier = ATR_SL_Multiplier;
   g_MinProfitPips = MinProfitPips;
   g_EnableTimeBasedExit = EnableTimeBasedExit;
   g_MaxPositionHoldBars = MaxPositionHoldBars;
   g_ExitBarMinute = InpExitBarMinute;
   g_EnableTrailingStop = EnableTrailingStop;
   g_TrailingStartPips = TrailingStartPips;
   g_TrailingStopPips = TrailingStopPips;
   g_EnableADXFilter = EnableADXFilter;
   g_ADX_Period = ADX_Period;
   g_ADX_Threshold = ADX_Threshold;
  }

bool LoadBacktestPredictions()
  {
   ArrayFree(g_backtest_predictions);
   g_backtest_prediction_idx = 0;

   // FIX: Added the FILE_COMMON flag to look in the shared terminal folder
   if(!FileIsExist(BACKTEST_PREDICTIONS_FILE, FILE_COMMON))
     {
      PrintFormat("FATAL ERROR: Backtest predictions file not found in Common/Files folder! Please place it there. File searched: %s", BACKTEST_PREDICTIONS_FILE);
      return false;
     }

   // FIX: Added the FILE_COMMON flag here as well
   int file_handle = FileOpen(BACKTEST_PREDICTIONS_FILE, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON);
   if(file_handle == INVALID_HANDLE)
     {
      PrintFormat("FATAL ERROR: Could not open backtest predictions file from Common/Files. Code: %d", GetLastError());
      return false;
     }

   FileReadString(file_handle); // Skip the header row

   int count = 0;
   while(!FileIsEnding(file_handle))
     {
      string line_parts[];
      string line = FileReadString(file_handle);
      if(StringSplit(line, ',', line_parts) < 4 + PREDICTION_STEPS)
         continue;

      ArrayResize(g_backtest_predictions, count + 1);
      
      g_backtest_predictions[count].timestamp = StringToTime(line_parts[0]);
      g_backtest_predictions[count].buy_prob = StringToDouble(line_parts[1]);
      g_backtest_predictions[count].sell_prob = StringToDouble(line_parts[2]);
      g_backtest_predictions[count].hold_prob = StringToDouble(line_parts[3]);
      for(int i=0; i<PREDICTION_STEPS; i++)
        {
         g_backtest_predictions[count].predicted_prices[i] = StringToDouble(line_parts[4+i]);
        }
      count++;
     }
   FileClose(file_handle);
   PrintFormat("Successfully loaded %d pre-computed predictions for backtesting from Common folder.", count);
   return(count > 0);
  }

bool FindPredictionForBar(datetime bar_time, BacktestPrediction &found_pred)
  {
   for(int i = g_backtest_prediction_idx; i < ArraySize(g_backtest_predictions); i++)
     {
      if(g_backtest_predictions[i].timestamp == bar_time)
        {
         found_pred = g_backtest_predictions[i];
         g_backtest_prediction_idx = i;
         return true;
        }
      if(g_backtest_predictions[i].timestamp > bar_time)
        {
         // We've gone past the time, no need to search further
         return false;
        }
     }
   return false;
  }
  
string GenerateRequestID()
  {
   MathSrand((int)GetTickCount());
   string id = (string)TimeLocal() + "_" + IntegerToString(MathRand());
   StringReplace(id, ":", "-");
   StringReplace(id, " ", "_");
   return id;
  }
  
bool SendToDaemonForClassification(const double &features[], double &buy_prob, double &sell_prob, double &hold_prob)
  {
   string request_id = GenerateRequestID();
   string filename = "request_" + request_id + ".json";
   string response_file = "response_" + request_id + ".json";
   string json = "{\r\n";
   json += "  \"request_id\": \"" + request_id + "\",\r\n";
   json += "  \"action\": \"predict_classification\",\r\n";
   json += "  \"features\": [";
   int features_total = ArraySize(features);
   for(int i = 0; i < features_total; i++)
     {
      json += DoubleToString(features[i], 8);
      if(i < features_total - 1)
         json += ", ";
     }
   json += "]\r\n}";
   string path = DATA_FOLDER + "\\" + filename;
   int file_handle = FileOpen(path, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(file_handle == INVALID_HANDLE) { PrintFormat("Error writing request: %s, Code: %d", path, GetLastError()); return false; }
   FileWriteString(file_handle, json);
   FileClose(file_handle);
   string response_path = DATA_FOLDER + "\\" + response_file;
   long start_time = GetTickCount();
   while(GetTickCount() - start_time < RequestTimeout)
     {
      Sleep(100);
      if(FileIsExist(response_path))
        {
         Sleep(50);
         int rfile = FileOpen(response_path, FILE_READ | FILE_TXT | FILE_ANSI);
         if(rfile == INVALID_HANDLE) return false;
         string content = FileReadString(rfile);
         FileClose(rfile);
         FileDelete(response_path);
         int buy_pos = StringFind(content, "\"buy_probability\""), sell_pos = StringFind(content, "\"sell_probability\""), hold_pos = StringFind(content, "\"hold_probability\"");
         if(buy_pos < 0 || sell_pos < 0 || hold_pos < 0) continue;
         string buy_str = StringSubstr(content, StringFind(content, ":", buy_pos) + 1);
         buy_str = StringSubstr(buy_str, 0, StringFind(buy_str, ","));
         StringTrimLeft(buy_str); StringTrimRight(buy_str); buy_prob = StringToDouble(buy_str);
         string sell_str = StringSubstr(content, StringFind(content, ":", sell_pos) + 1);
         sell_str = StringSubstr(sell_str, 0, StringFind(sell_str, ","));
         StringTrimLeft(sell_str); StringTrimRight(sell_str); sell_prob = StringToDouble(sell_str);
         string hold_str = StringSubstr(content, StringFind(content, ":", hold_pos) + 1);
         hold_str = StringSubstr(hold_str, 0, StringFind(hold_str, "}"));
         StringTrimLeft(hold_str); StringTrimRight(hold_str); hold_prob = StringToDouble(hold_str);
         return true;
        }
     }
   return false;
  }
  
bool SendToDaemonForRegression(const double &features[], double &predictions[])
  {
   string request_id = GenerateRequestID();
   string filename = "request_" + request_id + ".json";
   string response_file = "response_" + request_id + ".json";
   string json = "{\r\n";
   json += "  \"request_id\": \"" + request_id + "\",\r\n";
   json += "  \"action\": \"predict_regression\",\r\n";
   json += "  \"features\": [";
   int features_total = ArraySize(features);
   for(int i = 0; i < features_total; i++)
     {
      json += DoubleToString(features[i], 8);
      if(i < features_total - 1) json += ", ";
     }
   json += "]\r\n}";
   string path = DATA_FOLDER + "\\" + filename;
   int file_handle = FileOpen(path, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(file_handle == INVALID_HANDLE) { PrintFormat("Error writing request: %s, Code: %d", path, GetLastError()); return false; }
   FileWriteString(file_handle, json);
   FileClose(file_handle);
   string response_path = DATA_FOLDER + "\\" + response_file;
   long start_time = GetTickCount();
   while(GetTickCount() - start_time < RequestTimeout)
     {
      Sleep(100);
      if(FileIsExist(response_path))
        {
         Sleep(50);
         int rfile = FileOpen(response_path, FILE_READ | FILE_TXT | FILE_ANSI);
         if(rfile == INVALID_HANDLE) return false;
         string content = FileReadString(rfile);
         FileClose(rfile);
         FileDelete(response_path);
         int prices_pos = StringFind(content, "\"predicted_prices\"");
         if(prices_pos < 0) continue;
         int start_bracket = StringFind(content, "[", prices_pos);
         int end_bracket = StringFind(content, "]", start_bracket);
         if(start_bracket < 0 || end_bracket < 0) continue;
         string prices_str = StringSubstr(content, start_bracket + 1, end_bracket - start_bracket - 1);
         string price_values[];
         if(StringSplit(prices_str, ',', price_values) == PREDICTION_STEPS)
           {
            for(int i = 0; i < PREDICTION_STEPS; i++)
              {
               StringTrimLeft(price_values[i]); StringTrimRight(price_values[i]);
               predictions[i] = StringToDouble(price_values[i]);
              }
            return true;
           }
        }
     }
   return false;
  }
  
double CalculateLotSize(double stopLossPrice, double entryPrice)
  {
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(accountBalance <= 0)
     {
      Print("Error: Account Balance is zero.");
      return 0.0;
     }
   double riskAmount = accountBalance * (g_RiskPercent / 100.0);
   double loss_for_one_lot = 0;
   ENUM_ORDER_TYPE orderType = (entryPrice > stopLossPrice) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(!OrderCalcProfit(orderType, _Symbol, 1.0, entryPrice, stopLossPrice, loss_for_one_lot))
     {
      PrintFormat("Error: OrderCalcProfit() failed. Code: %d", GetLastError());
      return 0.0;
     }
   double loss_for_one_lot_abs = MathAbs(loss_for_one_lot);
   if(loss_for_one_lot_abs <= 0)
     {
      Print("Error: Loss for 1.0 lot is zero. Check SL.");
      return 0.0;
     }
   double lotSize = riskAmount / loss_for_one_lot_abs;
   double minVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), maxVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lotSize = MathFloor(lotSize / volStep) * volStep;
   if(lotSize < minVolume)
      lotSize = minVolume;
   if(lotSize > maxVolume)
      lotSize = maxVolume;
   return NormalizeDouble(lotSize, 2);
  }

void UpdateChartDisplay()
  {
   if(!EnablePricePredictionDisplay)
     {
      Comment("");
      return;
     }
   string display_text = "--- LSTM Price Prediction ---\n";
   for(int i = 0; i < PREDICTION_STEPS; i++)
     {
      string price_str = (g_last_predictions[i] == 0) ? "Calculating..." : DoubleToString(g_last_predictions[i], _Digits);
      display_text += StringFormat("H+%d Price: %s", i + 1, price_str);
      if(g_total_predictions[i] > 0)
        {
         g_accuracy_pct[i] = ((double)g_total_hits[i] / (double)g_total_predictions[i]) * 100.0;
         display_text += StringFormat(" (Accuracy: %.1f%%, N=%d)\n", g_accuracy_pct[i], g_total_predictions[i]);
        }
      else
        {
         display_text += " (Accuracy: N/A)\n";
        }
     }
   Comment(display_text);
  }

void CheckPastPredictionAccuracy()
  {
   if(!EnablePricePredictionDisplay || ArraySize(g_pending_predictions) == 0)
      return;
   double bar_high = iHigh(_Symbol, PERIOD_H1, 1), bar_low = iLow(_Symbol, PERIOD_H1, 1);
   datetime bar_time = (datetime)SeriesInfoInteger(_Symbol, PERIOD_H1, SERIES_LASTBAR_DATE);
   for(int i = ArraySize(g_pending_predictions) - 1; i >= 0; i--)
     {
      PendingPrediction pred = g_pending_predictions[i];
      bool is_hit = false, is_expired = false;
      if(pred.direction == DIR_BULLISH && pred.target_price <= bar_high)
         is_hit = true;
      else
         if(pred.direction == DIR_BEARISH && pred.target_price >= bar_low)
            is_hit = true;
      if(bar_time >= pred.end_time)
         is_expired = true;
      if(is_hit || is_expired)
        {
         g_total_predictions[pred.step]++;
         if(is_hit)
            g_total_hits[pred.step]++;
         ArrayRemove(g_pending_predictions, i, 1);
        }
     }
  }

void EnsureDataFolderExists() { if(!FolderCreate(DATA_FOLDER)) { PrintFormat("Warning: Could not create folder 'MQL5\\Files\\%s'.", DATA_FOLDER); } }

void ManageTrailingStop()
  {
   if(!PositionSelect(_Symbol))
      return;
   double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN), currentSL = PositionGetDouble(POSITION_SL);
   long positionType = PositionGetInteger(POSITION_TYPE);
   MqlTick latest_tick;
   if(!SymbolInfoTick(_Symbol, latest_tick))
      return;
   double pips_to_points = (_Digits == 3 || _Digits == 5) ? _Point * 10 : _Point;
   if(positionType == POSITION_TYPE_BUY)
     {
      if((latest_tick.bid - entryPrice) > (g_TrailingStartPips * pips_to_points))
        {
         double newSL = latest_tick.bid - (g_TrailingStopPips * pips_to_points);
         if(newSL > currentSL || currentSL == 0)
            trade.PositionModify(_Symbol, newSL, PositionGetDouble(POSITION_TP));
        }
     }
   else
      if(positionType == POSITION_TYPE_SELL)
        {
         if((entryPrice - latest_tick.ask) > (g_TrailingStartPips * pips_to_points))
           {
            double newSL = latest_tick.ask + (g_TrailingStopPips * pips_to_points);
            if(newSL < currentSL || currentSL == 0)
               trade.PositionModify(_Symbol, newSL, PositionGetDouble(POSITION_TP));
           }
        }
  }

//+------------------------------------------------------------------+
//| MQL5 Main Event Handlers
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("=== Siobhan EA v1.42 (Backtest Enabled & Fixed) Initializing ===");
   
   InitializeParameters();

   if(MQLInfoInteger(MQL_TESTER))
     {
      Print("Strategy Tester mode detected. Attempting to load pre-computed predictions...");
      if(!LoadBacktestPredictions())
        {
         return(INIT_FAILED);
        }
     }
   else
     {
      Print("Live/Demo trading mode detected. Will use Python daemon.");
      LoadLearnedParameters();
     }
   
   if(g_ExitBarMinute < 0 || g_ExitBarMinute > 59)
     {
      Print("Error: InpExitBarMinute must be 0-59. Defaulting to 58.");
      g_ExitBarMinute = 58;
     }
   EnsureDataFolderExists();
   SymbolSelect(Symbol_EURJPY, true);
   SymbolSelect(Symbol_USDJPY, true);
   SymbolSelect(Symbol_GBPUSD, true);
   SymbolSelect(Symbol_EURGBP, true);
   SymbolSelect(Symbol_USDCAD, true);
   SymbolSelect(Symbol_USDCHF, true);

   atr_handle = iATR(_Symbol, PERIOD_H1, g_ATR_Period);
   macd_handle = iMACD(_Symbol, PERIOD_H1, 12, 26, 9, PRICE_CLOSE);
   rsi_handle = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
   stoch_handle = iStochastic(_Symbol, PERIOD_H1, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
   cci_handle = iCCI(_Symbol, PERIOD_H1, 20, PRICE_TYPICAL);
   adx_handle = iADX(_Symbol, PERIOD_H1, g_ADX_Period);
   
   if(atr_handle==INVALID_HANDLE || macd_handle==INVALID_HANDLE || rsi_handle==INVALID_HANDLE || stoch_handle==INVALID_HANDLE || cci_handle==INVALID_HANDLE || adx_handle==INVALID_HANDLE)
     { Print("Error: Failed to create one or more indicator handles."); return(INIT_FAILED); }
   ArrayInitialize(g_last_predictions, 0.0);
   ArrayInitialize(g_accuracy_pct, 0.0);
   ArrayInitialize(g_total_hits, 0);
   ArrayInitialize(g_total_predictions, 0);
   ArrayFree(g_pending_predictions);
   UpdateChartDisplay();
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   IndicatorRelease(atr_handle);
   IndicatorRelease(macd_handle);
   IndicatorRelease(rsi_handle);
   IndicatorRelease(stoch_handle);
   IndicatorRelease(cci_handle);
   IndicatorRelease(adx_handle);
   Comment("");
  }
//+------------------------------------------------------------------+
void OnTick()
  {
   if(PositionsTotal() > 0 && PositionSelect(_Symbol))
     {
      if(UseMarketOrderForTP && g_active_trade_target_price > 0)
        {
         MqlTick latest_tick;
         if(SymbolInfoTick(_Symbol, latest_tick))
           {
            if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && latest_tick.bid >= g_active_trade_target_price)
              { PrintFormat("Closing by market, TP hit: %d", (long)PositionGetInteger(POSITION_TICKET)); trade.PositionClose(_Symbol); g_active_trade_target_price = 0; return; }
            if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && latest_tick.ask <= g_active_trade_target_price)
              { PrintFormat("Closing by market, TP hit: %d", (long)PositionGetInteger(POSITION_TICKET)); trade.PositionClose(_Symbol); g_active_trade_target_price = 0; return; }
           }
        }
      if(g_EnableTimeBasedExit)
        {
         datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
         datetime deadline = open_time + (datetime)(g_MaxPositionHoldBars * PeriodSeconds(PERIOD_H1));
         if(TimeCurrent() >= deadline)
           {
            MqlDateTime now;
            TimeToStruct(TimeCurrent(), now);
            if(now.min >= g_ExitBarMinute)
              { PrintFormat("Closing by time limit: %d", (long)PositionGetInteger(POSITION_TICKET)); trade.PositionClose(_Symbol); g_active_trade_target_price = 0; return; }
           }
        }
      if(g_EnableTrailingStop)
         ManageTrailingStop();
      return;
     }

   static datetime last_bar_time = 0;
   datetime current_bar_time = (datetime)SeriesInfoInteger(_Symbol, PERIOD_H1, SERIES_LASTBAR_DATE);
   if(current_bar_time == last_bar_time)
      return;
   last_bar_time = current_bar_time;

   g_active_trade_target_price = 0;
   CheckPastPredictionAccuracy();

   if(TradingLogicMode == MODE_TRADING_DISABLED)
      return;
   
   // --- PREDICTION LOGIC ---
   double predicted_prices[PREDICTION_STEPS];
   double buy_prob = 0, sell_prob = 0, hold_prob = 1;
   bool got_prediction = false;
   
   if(MQLInfoInteger(MQL_TESTER))
     {
      // --- BACKTESTING MODE ---
      datetime bar_to_check = iTime(_Symbol, PERIOD_H1, 1);
      BacktestPrediction current_pred;
      if(FindPredictionForBar(bar_to_check, current_pred))
        {
         buy_prob = current_pred.buy_prob;
         sell_prob = current_pred.sell_prob;
         hold_prob = current_pred.hold_prob;
         for(int i=0; i<PREDICTION_STEPS; i++)
           {
            predicted_prices[i] = current_pred.predicted_prices[i];
            g_last_predictions[i] = predicted_prices[i];
           }
         got_prediction = true;
        }
     }
   else
     {
      // --- LIVE TRADING MODE ---
      double features[SEQ_LEN * FEATURE_COUNT];
      int data_needed = SEQ_LEN + 30;
      double eurusd_c[], eurjpy_c[], usdjpy_c[], gbpusd_c[], eurgbp_c[], usdcad_c[], usdchf_c[];
      double eurusd_h[], eurusd_l[];
      long eurusd_vol[];
      double macd_buf[], rsi_buf[], stoch_buf[], cci_buf[];
      if(CopyClose(_Symbol, PERIOD_H1, 0, data_needed, eurusd_c) < data_needed || CopyHigh(_Symbol, PERIOD_H1, 0, data_needed, eurusd_h) < data_needed || CopyLow(_Symbol, PERIOD_H1, 0, data_needed, eurusd_l) < data_needed || CopyTickVolume(_Symbol, PERIOD_H1, 0, data_needed, eurusd_vol) < data_needed) return;
      if(CopyBuffer(macd_handle, 0, 0, data_needed, macd_buf) < data_needed || CopyBuffer(rsi_handle, 0, 0, data_needed, rsi_buf) < data_needed || CopyBuffer(stoch_handle, 0, 0, data_needed, stoch_buf) < data_needed || CopyBuffer(cci_handle, 0, 0, data_needed, cci_buf) < data_needed) return;
      if(CopyClose(Symbol_EURJPY, PERIOD_H1, 0, data_needed, eurjpy_c) < data_needed || CopyClose(Symbol_USDJPY, PERIOD_H1, 0, data_needed, usdjpy_c) < data_needed || CopyClose(Symbol_GBPUSD, PERIOD_H1, 0, data_needed, gbpusd_c) < data_needed || CopyClose(Symbol_EURGBP, PERIOD_H1, 0, data_needed, eurgbp_c) < data_needed || CopyClose(Symbol_USDCAD, PERIOD_H1, 0, data_needed, usdcad_c) < data_needed || CopyClose(Symbol_USDCHF, PERIOD_H1, 0, data_needed, usdchf_c) < data_needed) return;
      int feature_index = 0;
      for(int i = 1; i <= SEQ_LEN; i++)
        {
         double eurusd_ret = (eurusd_c[i] / eurusd_c[i + 1]) - 1.0, eurjpy_ret = (eurjpy_c[i] / eurjpy_c[i + 1]) - 1.0, usdjpy_ret = (usdjpy_c[i] / usdjpy_c[i + 1]) - 1.0, gbpusd_ret = (gbpusd_c[i] / gbpusd_c[i + 1]) - 1.0, eurgbp_ret = (eurgbp_c[i] / eurgbp_c[i + 1]) - 1.0, usdcad_ret = (usdcad_c[i] / usdcad_c[i + 1]) - 1.0, usdchf_ret = (usdchf_c[i] / usdchf_c[i + 1]) - 1.0;
         MqlDateTime time_struct;
         TimeToStruct(iTime(_Symbol, PERIOD_H1, i), time_struct);
         features[feature_index++] = eurusd_ret; features[feature_index++] = (double)eurusd_vol[i]; features[feature_index++] = (eurusd_h[i] - eurusd_l[i]);
         features[feature_index++] = macd_buf[i]; features[feature_index++] = rsi_buf[i]; features[feature_index++] = stoch_buf[i]; features[feature_index++] = cci_buf[i];
         features[feature_index++] = (double)time_struct.hour; features[feature_index++] = (double)time_struct.day_of_week;
         features[feature_index++] = (usdjpy_ret + usdcad_ret + usdchf_ret) - (eurusd_ret + gbpusd_ret);
         features[feature_index++] = eurusd_ret + eurjpy_ret + eurgbp_ret;
         features[feature_index++] = -(eurjpy_ret + usdjpy_ret);
        }

      bool got_regression_pred = false;
      bool got_classification_pred = false;

      if(TradingLogicMode != MODE_CLASSIFICATION_ONLY)
        {
         got_regression_pred = SendToDaemonForRegression(features, predicted_prices);
        }
      if(TradingLogicMode != MODE_REGRESSION_ONLY)
        {
         got_classification_pred = SendToDaemonForClassification(features, buy_prob, sell_prob, hold_prob);
        }
        
      got_prediction = got_regression_pred || got_classification_pred;
      if(got_regression_pred) { for(int i=0; i<PREDICTION_STEPS; i++) g_last_predictions[i] = predicted_prices[i]; }
     }
     
   if(EnablePricePredictionDisplay)
      UpdateChartDisplay();
   
   if(!got_prediction) 
      return; 

   // --- TRADING LOGIC (Now runs for both live and backtest) ---
   if(g_EnableADXFilter)
     {
      double adx_buffer[];
      if(CopyBuffer(adx_handle, 0, 1, 1, adx_buffer) < 1) return;
      if(adx_buffer[0] < g_ADX_Threshold) return;
     }

   MqlTick latest_tick;
   if(!SymbolInfoTick(_Symbol, latest_tick)) return;
   double atr_value_arr[];
   if(CopyBuffer(atr_handle, 0, 1, 1, atr_value_arr) < 1) return;
   double atr_value = atr_value_arr[0];
   double pips_to_points = (_Digits == 3 || _Digits == 5) ? _Point * 10 : _Point;

   switch(TradingLogicMode)
     {
      case MODE_CLASSIFICATION_ONLY:
         {
            if(buy_prob > g_SignalThresholdProbability) { /* Your Buy Logic for Classification Only would go here */ }
            if(sell_prob > g_SignalThresholdProbability) { /* Your Sell Logic for Classification Only would go here */ }
            break;
         }

      case MODE_REGRESSION_ONLY:
      case MODE_COMBINED:
         {
            double target_price = predicted_prices[TakeProfitTargetBar];
            double spread_points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
            double min_profit_points = g_MinProfitPips * pips_to_points;

            int bullish_steps = 0, bearish_steps = 0;
            for(int i = 0; i < PREDICTION_STEPS; i++)
              {
               if(predicted_prices[i] > latest_tick.ask) bullish_steps++;
               if(predicted_prices[i] < latest_tick.bid) bearish_steps++;
              }

            if(bullish_steps >= g_RequiredConsistentSteps && bullish_steps > bearish_steps)
              {
               if((target_price - latest_tick.ask) > (min_profit_points + spread_points))
                 {
                  double sl;
                  if(StopLossMode == SL_STATIC_PIPS) sl = latest_tick.ask - (g_StaticStopLossPips * pips_to_points);
                  else sl = latest_tick.ask - (atr_value * g_ATR_SL_Multiplier);
                  bool valid = (StopLossMode == SL_STATIC_PIPS) ? true : (sl < latest_tick.ask && (target_price-latest_tick.ask)/(latest_tick.ask-sl) >= g_MinimumRiskRewardRatio);
                  if(valid)
                    {
                     if(TradingLogicMode == MODE_COMBINED && (buy_prob < sell_prob)) break;
                     double lots = CalculateLotSize(sl, latest_tick.ask);
                     double tp = UseMarketOrderForTP ? 0 : target_price;
                     if(lots > 0 && trade.Buy(lots, _Symbol, latest_tick.ask, sl, tp, "LSTM Regress Buy"))
                       {
                        if(UseMarketOrderForTP) g_active_trade_target_price = target_price;
                        if(!MQLInfoInteger(MQL_TESTER)) {
                           for(int step=0; step<PREDICTION_STEPS; step++){
                              PendingPrediction pred; pred.target_price=predicted_prices[step]; pred.start_time=TimeCurrent(); pred.end_time=TimeCurrent() + (AccuracyLookaheadBars*PeriodSeconds(PERIOD_H1));
                              pred.direction=DIR_BULLISH; pred.step=step; int sz=ArraySize(g_pending_predictions); ArrayResize(g_pending_predictions, sz+1); g_pending_predictions[sz]=pred;
                           }
                        }
                        return;
                       }
                    }
                 }
              }
            
            if(bearish_steps >= g_RequiredConsistentSteps && bearish_steps > bullish_steps)
              {
               if((latest_tick.bid - target_price) > (min_profit_points + spread_points))
                 {
                  double sl;
                  if(StopLossMode == SL_STATIC_PIPS) sl = latest_tick.bid + (g_StaticStopLossPips * pips_to_points);
                  else sl = latest_tick.bid + (atr_value * g_ATR_SL_Multiplier);
                  bool valid = (StopLossMode == SL_STATIC_PIPS) ? true : (sl > latest_tick.bid && (latest_tick.bid-target_price)/(sl-latest_tick.bid) >= g_MinimumRiskRewardRatio);
                  if(valid)
                    {
                     if(TradingLogicMode == MODE_COMBINED && (sell_prob < buy_prob)) break;
                     double lots = CalculateLotSize(sl, latest_tick.bid);
                     double tp = UseMarketOrderForTP ? 0 : target_price;
                     if(lots > 0 && trade.Sell(lots, _Symbol, latest_tick.bid, sl, tp, "LSTM Regress Sell"))
                       {
                        if(UseMarketOrderForTP) g_active_trade_target_price = target_price;
                        if(!MQLInfoInteger(MQL_TESTER)) {
                           for(int step=0; step<PREDICTION_STEPS; step++){
                              PendingPrediction pred; pred.target_price=predicted_prices[step]; pred.start_time=TimeCurrent(); pred.end_time=TimeCurrent() + (AccuracyLookaheadBars*PeriodSeconds(PERIOD_H1));
                              pred.direction=DIR_BEARISH; pred.step=step; int sz=ArraySize(g_pending_predictions); ArrayResize(g_pending_predictions, sz+1); g_pending_predictions[sz]=pred;
                           }
                        }
                        return;
                       }
                    }
                 }
              }
            break;
         }
     }
  }

//+------------------------------------------------------------------+
//| OnTester: Called at the end of a backtest/optimization pass.
//+------------------------------------------------------------------+
double OnTester()
  {
   double history_profits[];
   HistorySelect(0, TimeCurrent());
   int deals = HistoryDealsTotal(), profit_count = 0;
   if(deals <= 1) return 0.0;
   ArrayResize(history_profits, deals);
   for(int i = 0; i < deals; i++)
     {
      long ticket = (long)HistoryDealGetTicket(i);
      if(ticket > 0 && HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         history_profits[profit_count++] = HistoryDealGetDouble(ticket, DEAL_PROFIT);
     }
   if(profit_count <= 1) return 0.0;
   ArrayResize(history_profits, profit_count);
   double mean_profit = MathMean(history_profits);
   double std_dev_profit = MathStandardDeviation(history_profits);
   if(std_dev_profit < 0.0001) return 0.0;
   double sharpe_ratio = mean_profit / std_dev_profit;
   double custom_criterion = sharpe_ratio * MathSqrt(profit_count);
   PrintFormat("OnTester Pass Complete: Trades=%d, Mean Profit=%.2f, StdDev=%.2f, Sharpe=%.3f, Custom Criterion=%.3f",
               profit_count, mean_profit, std_dev_profit, sharpe_ratio, custom_criterion);
   return custom_criterion;
  }
//+------------------------------------------------------------------+