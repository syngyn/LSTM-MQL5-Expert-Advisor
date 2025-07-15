//+------------------------------------------------------------------+
//|                                                        Siobhan.mq5 |
//|                                         Copyright 2025, Jason Rusk |
//|  Dedicated to my late wife who always supported my trading passion |
//|                                                                    |
//+------------------------------------------------------------------+

#property copyright "Jason.W.Rusk@gmail.com 2025"
#property version   "1.62" // Fixed compiler errors from input grouping

#include <Trade/Trade.mqh>
#include <Files/File.mqh>
#include <stdlib.mqh>
#include <Math/Stat/Math.mqh>

// --- TYPE DEFINITIONS (MUST BE DECLARED BEFORE INPUTS) ---
enum ENUM_TRADING_MODE { MODE_TRADING_DISABLED, MODE_CLASSIFICATION_ONLY, MODE_REGRESSION_ONLY, MODE_COMBINED };
enum ENUM_STOP_LOSS_MODE { SL_ATR_BASED, SL_STATIC_PIPS };
enum ENUM_TAKE_PROFIT_MODE { TP_REGRESSION_TARGET, TP_ATR_MULTIPLE, TP_STATIC_PIPS };
enum ENUM_TARGET_BAR 
{ 
    H_PLUS_1=0, H_PLUS_2, H_PLUS_3, H_PLUS_4, H_PLUS_5, H_PLUS_6,
    H_PLUS_7, H_PLUS_8, H_PLUS_9, H_PLUS_10, H_PLUS_11, H_PLUS_12,
    H_PLUS_13, H_PLUS_14, H_PLUS_15, H_PLUS_16, H_PLUS_17, H_PLUS_18,
    H_PLUS_19, H_PLUS_20, H_PLUS_21, H_PLUS_22, H_PLUS_23, H_PLUS_24
};


// --- INPUT PARAMETERS ---
input group    "Main Settings"
input ENUM_TRADING_MODE TradingLogicMode = MODE_REGRESSION_ONLY;
input bool              EnablePricePredictionDisplay = true;
input ENUM_TARGET_BAR TakeProfitTargetBar = H_PLUS_12;

input group    "Risk & Position Management"
input ENUM_STOP_LOSS_MODE   StopLossMode = SL_ATR_BASED;
input ENUM_TAKE_PROFIT_MODE TakeProfitMode = TP_REGRESSION_TARGET;
input bool UseMarketOrderForTP = false;
input double RiskPercent = 3.0;
input double MinimumRiskRewardRatio = 1.5;
input int    StaticStopLossPips = 300;
input int    StaticTakeProfitPips = 400;
input int    ATR_Period = 14;
input double ATR_SL_Multiplier = 1.5;
input double ATR_TP_Multiplier = 2.0;
input double MinProfitPips = 10.0;
input bool   EnableTimeBasedExit   = true;
input int    MaxPositionHoldBars   = 12;
input int    InpExitBarMinute      = 58;
input bool   EnableTrailingStop = true;
input double TrailingStartPips = 12.0;
input double TrailingStopPips = 3.0;

input group    "Confidence & Filters"
input double MinimumConfidence = 0.65;
input int    RequiredConsistentSteps = 16;
input bool   EnableADXFilter = true;
input int    ADX_Period = 14;
input int    ADX_Threshold = 25;

input group    "Model & Data Settings"
input int    AccuracyLookaheadBars = 24;
input string Symbol_EURJPY = "EURJPY", Symbol_USDJPY = "USDJPY", Symbol_GBPUSD = "GBPUSD";
input string Symbol_EURGBP = "EURGBP", Symbol_USDCAD = "USDCAD", Symbol_USDCHF = "USDCHF";
input int    RequestTimeout = 5000;

// --- Constants ---
#define PREDICTION_STEPS 24
#define SEQ_LEN 20
#define FEATURE_COUNT 12
#define DATA_FOLDER "LSTM_Trading\\data"
#define GUI_PREFIX "SiobhanGUI_"
#define BACKTEST_PREDICTIONS_FILE "backtest_predictions.csv"

// --- Global Handles & Variables ---
int atr_handle, macd_handle, rsi_handle, stoch_handle, cci_handle, adx_handle;
CTrade trade;
enum ENUM_PREDICTION_DIRECTION { DIR_BULLISH, DIR_BEARISH, DIR_NEUTRAL };
struct PendingPrediction { double target_price; datetime start_time, end_time; ENUM_PREDICTION_DIRECTION direction; int step; };
PendingPrediction g_pending_predictions[];
double g_last_predictions[PREDICTION_STEPS];
double g_accuracy_pct[PREDICTION_STEPS];
int    g_total_hits[PREDICTION_STEPS], g_total_predictions[PREDICTION_STEPS];
double g_active_trade_target_price = 0;

struct BacktestPrediction { datetime timestamp; double buy_prob, sell_prob, hold_prob; double predicted_prices[PREDICTION_STEPS]; };
BacktestPrediction g_backtest_predictions[];
int g_backtest_prediction_idx = 0;

// --- Global Parameters ---
double g_RiskPercent, g_MinimumRiskRewardRatio, g_ATR_SL_Multiplier, g_ATR_TP_Multiplier, g_MinProfitPips, g_TrailingStartPips, g_TrailingStopPips, g_MinimumConfidence;
int    g_RequiredConsistentSteps, g_StaticStopLossPips, g_StaticTakeProfitPips, g_ATR_Period, g_MaxPositionHoldBars, g_ExitBarMinute, g_ADX_Period, g_ADX_Threshold;
bool   g_EnableTimeBasedExit, g_EnableTrailingStop, g_EnableADXFilter;

//+------------------------------------------------------------------+
//| --- GUI PANEL FUNCTIONS ---
//+------------------------------------------------------------------+
void CreateDisplayPanel()
  {
   if(!EnablePricePredictionDisplay) return;
   string bg_name = GUI_PREFIX + "background";
   ObjectCreate(0, bg_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, bg_name, OBJPROP_XDISTANCE, 5);
   ObjectSetInteger(0, bg_name, OBJPROP_YDISTANCE, 20);
   ObjectSetInteger(0, bg_name, OBJPROP_XSIZE, 240);
   ObjectSetInteger(0, bg_name, OBJPROP_YSIZE, 15 + (PREDICTION_STEPS * 14) + 5);
   ObjectSetInteger(0, bg_name, OBJPROP_BGCOLOR, C'20,20,40');
   ObjectSetInteger(0, bg_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, bg_name, OBJPROP_BACK, true);

   string title_name = GUI_PREFIX + "title";
   ObjectCreate(0, title_name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, title_name, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, title_name, OBJPROP_YDISTANCE, 25);
   ObjectSetString(0, title_name, OBJPROP_TEXT, "Siobhan LSTM Prediction (H1)");
   ObjectSetInteger(0, title_name, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, title_name, OBJPROP_FONTSIZE, 9);

   int y_pos = 40;
   for(int i = 0; i < PREDICTION_STEPS; i++)
     {
      string hour_label_name = GUI_PREFIX + "hour_" + (string)i;
      ObjectCreate(0, hour_label_name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, hour_label_name, OBJPROP_XDISTANCE, 15);
      ObjectSetInteger(0, hour_label_name, OBJPROP_YDISTANCE, y_pos);
      ObjectSetString(0, hour_label_name, OBJPROP_TEXT, StringFormat("H+%d:", i + 1));
      ObjectSetInteger(0, hour_label_name, OBJPROP_COLOR, clrSilver);
      ObjectSetInteger(0, hour_label_name, OBJPROP_FONTSIZE, 8);

      string price_label_name = GUI_PREFIX + "price_" + (string)i;
      ObjectCreate(0, price_label_name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, price_label_name, OBJPROP_XDISTANCE, 55);
      ObjectSetInteger(0, price_label_name, OBJPROP_YDISTANCE, y_pos);
      ObjectSetString(0, price_label_name, OBJPROP_TEXT, "Calculating...");
      ObjectSetInteger(0, price_label_name, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, price_label_name, OBJPROP_FONTSIZE, 8);
      
      string acc_label_name = GUI_PREFIX + "acc_" + (string)i;
      ObjectCreate(0, acc_label_name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, acc_label_name, OBJPROP_XDISTANCE, 140);
      ObjectSetInteger(0, acc_label_name, OBJPROP_YDISTANCE, y_pos);
      ObjectSetString(0, acc_label_name, OBJPROP_TEXT, "Acc: N/A");
      ObjectSetInteger(0, acc_label_name, OBJPROP_COLOR, clrGray);
      ObjectSetInteger(0, acc_label_name, OBJPROP_FONTSIZE, 8);
      y_pos += 14;
     }
   ChartRedraw();
  }

void UpdateDisplayPanel()
  {
   if(!EnablePricePredictionDisplay) return;
   for(int i = 0; i < PREDICTION_STEPS; i++)
     {
      string price_label_name = GUI_PREFIX + "price_" + (string)i;
      string price_text = (g_last_predictions[i] == 0) ? "Calculating..." : DoubleToString(g_last_predictions[i], _Digits);
      ObjectSetString(0, price_label_name, OBJPROP_TEXT, price_text);
      string acc_label_name = GUI_PREFIX + "acc_" + (string)i;
      string acc_text = "Acc: N/A";
      if(g_total_predictions[i] > 0)
        {
         g_accuracy_pct[i] = ((double)g_total_hits[i] / (double)g_total_predictions[i]) * 100.0;
         acc_text = StringFormat("Acc: %.1f%% (N=%d)", g_accuracy_pct[i], g_total_predictions[i]);
        }
      ObjectSetString(0, acc_label_name, OBJPROP_TEXT, acc_text);
     }
   ChartRedraw();
  }
  
void DeleteDisplayPanel() { ObjectsDeleteAll(0, GUI_PREFIX); ChartRedraw(); }

//+------------------------------------------------------------------+
//| --- CORE HELPER FUNCTIONS ---
//+------------------------------------------------------------------+
bool JsonGetValue(const string &json_string, const string &key, double &out_value)
  {
   string search_key = "\"" + key + "\"";
   int key_pos = StringFind(json_string, search_key); if(key_pos < 0) return false;
   int colon_pos = StringFind(json_string, ":", key_pos); if(colon_pos < 0) return false;
   int next_comma_pos = StringFind(json_string, ",", colon_pos);
   int next_brace_pos = StringFind(json_string, "}", colon_pos);
   int end_pos = (next_comma_pos > 0 && (next_brace_pos < 0 || next_comma_pos < next_brace_pos)) ? next_comma_pos : next_brace_pos;
   if(end_pos < 0) end_pos = StringLen(json_string);
   string value_str = StringSubstr(json_string, colon_pos + 1, end_pos - (colon_pos + 1));
   StringTrimLeft(value_str); StringTrimRight(value_str);
   out_value = StringToDouble(value_str);
   return true;
  }
  
void InitializeParameters()
  {
   g_RiskPercent = RiskPercent; g_MinimumRiskRewardRatio = MinimumRiskRewardRatio;
   g_RequiredConsistentSteps = RequiredConsistentSteps; g_StaticStopLossPips = StaticStopLossPips;
   g_StaticTakeProfitPips = StaticTakeProfitPips; g_ATR_Period = ATR_Period;
   g_ATR_SL_Multiplier = ATR_SL_Multiplier; g_ATR_TP_Multiplier = ATR_TP_Multiplier;
   g_MinProfitPips = MinProfitPips; g_EnableTimeBasedExit = EnableTimeBasedExit;
   g_MaxPositionHoldBars = MaxPositionHoldBars; g_ExitBarMinute = InpExitBarMinute;
   g_EnableTrailingStop = EnableTrailingStop; g_TrailingStartPips = TrailingStartPips;
   g_TrailingStopPips = TrailingStopPips; g_EnableADXFilter = EnableADXFilter;
   g_ADX_Period = ADX_Period; g_ADX_Threshold = ADX_Threshold;
   g_MinimumConfidence = MinimumConfidence;
  }
  
bool LoadBacktestPredictions()
  {
   ArrayFree(g_backtest_predictions); g_backtest_prediction_idx = 0;
   if(!FileIsExist(BACKTEST_PREDICTIONS_FILE, FILE_COMMON)) { PrintFormat("FATAL: Backtest file not found: %s", BACKTEST_PREDICTIONS_FILE); return false; }
   int file_handle = FileOpen(BACKTEST_PREDICTIONS_FILE, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON);
   if(file_handle == INVALID_HANDLE) { PrintFormat("FATAL: Could not open backtest file. Code: %d", GetLastError()); return false; }
   FileReadString(file_handle);
   int count = 0;
   while(!FileIsEnding(file_handle))
     {
      string line_parts[]; string line = FileReadString(file_handle);
      if(StringSplit(line, ';', line_parts) < 4 + PREDICTION_STEPS) continue;
      ArrayResize(g_backtest_predictions, count + 1);
      g_backtest_predictions[count].timestamp = StringToTime(line_parts[0]);
      g_backtest_predictions[count].buy_prob = StringToDouble(line_parts[1]);
      g_backtest_predictions[count].sell_prob = StringToDouble(line_parts[2]);
      g_backtest_predictions[count].hold_prob = StringToDouble(line_parts[3]);
      for(int i=0; i<PREDICTION_STEPS; i++) g_backtest_predictions[count].predicted_prices[i] = StringToDouble(line_parts[4+i]);
      count++;
     }
   FileClose(file_handle);
   PrintFormat("Loaded %d pre-computed (%d-step) predictions.", count, PREDICTION_STEPS);
   return(count > 0);
  }

bool FindPredictionForBar(datetime bar_time, BacktestPrediction &found_pred)
  {
   for(int i = g_backtest_prediction_idx; i < ArraySize(g_backtest_predictions); i++)
     {
      if(g_backtest_predictions[i].timestamp == bar_time) { found_pred = g_backtest_predictions[i]; g_backtest_prediction_idx = i; return true; }
      if(g_backtest_predictions[i].timestamp > bar_time) return false;
     }
   return false;
  }
  
string GenerateRequestID() { MathSrand((int)GetTickCount()); string id = (string)TimeLocal() + "_" + IntegerToString(MathRand()); StringReplace(id, ":", "-"); StringReplace(id, " ", "_"); return id; }

bool SendToDaemon(const string &action, const double &features[], double &out_data[])
  {
   string request_id = GenerateRequestID();
   string filename = "request_" + request_id + ".json";
   string response_file = "response_" + request_id + ".json";
   string json = "{\r\n  \"request_id\": \"" + request_id + "\",\r\n  \"action\": \"" + action + "\",\r\n  \"features\": [";
   for(int i = 0; i < ArraySize(features); i++) json += DoubleToString(features[i], 8) + (i < ArraySize(features) - 1 ? ", " : "");
   json += "]\r\n}";
   
   int file_handle = FileOpen(DATA_FOLDER + "\\" + filename, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(file_handle == INVALID_HANDLE) { PrintFormat("Error writing request: %s, Code: %d", DATA_FOLDER + "\\" + filename, GetLastError()); return false; }
   FileWriteString(file_handle, json); FileClose(file_handle);
   
   long start_time = GetTickCount();
   while(GetTickCount() - start_time < RequestTimeout)
     {
      Sleep(100);
      if(FileIsExist(DATA_FOLDER + "\\" + response_file))
        {
         Sleep(50);
         int rfile = FileOpen(DATA_FOLDER + "\\" + response_file, FILE_READ | FILE_TXT | FILE_ANSI);
         if(rfile == INVALID_HANDLE) continue;
         string content = FileReadString(rfile);
         FileClose(rfile); FileDelete(DATA_FOLDER + "\\" + response_file);
         
         if(action == "predict_regression")
           {
            int prices_pos = StringFind(content, "\"predicted_prices\""); if(prices_pos < 0) continue;
            int start_bracket = StringFind(content, "[", prices_pos); int end_bracket = StringFind(content, "]", start_bracket);
            if(start_bracket < 0 || end_bracket < 0) continue;
            string prices_str = StringSubstr(content, start_bracket + 1, end_bracket - start_bracket - 1);
            string price_values[];
            if(StringSplit(prices_str, ',', price_values) == ArraySize(out_data))
              {
               for(int i = 0; i < ArraySize(out_data); i++) { StringTrimLeft(price_values[i]); StringTrimRight(price_values[i]); out_data[i] = StringToDouble(price_values[i]); }
               return true;
              }
           }
        }
     }
   return false;
  }

double CalculateLotSize(double stopLossPrice, double entryPrice)
  {
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE); if(accountBalance <= 0) return 0.0;
   double riskAmount = accountBalance * g_RiskPercent / 100.0;
   double loss_for_one_lot = 0;
   ENUM_ORDER_TYPE orderType = (entryPrice > stopLossPrice) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(!OrderCalcProfit(orderType, _Symbol, 1.0, entryPrice, stopLossPrice, loss_for_one_lot)) return 0.0;
   double loss_for_one_lot_abs = MathAbs(loss_for_one_lot); if(loss_for_one_lot_abs <= 0) return 0.0;
   double lotSize = riskAmount / loss_for_one_lot_abs;
   double minVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), maxVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lotSize = MathFloor(lotSize / volStep) * volStep;
   return(NormalizeDouble(fmin(maxVolume, fmax(minVolume, lotSize)), 2));
  }

void CheckPastPredictionAccuracy()
  {
   if(!EnablePricePredictionDisplay || ArraySize(g_pending_predictions) == 0) return;
   double bar_high = iHigh(_Symbol, PERIOD_H1, 1), bar_low = iLow(_Symbol, PERIOD_H1, 1);
   datetime bar_time = iTime(_Symbol, PERIOD_H1, 0);
   for(int i = ArraySize(g_pending_predictions) - 1; i >= 0; i--)
     {
      PendingPrediction pred = g_pending_predictions[i];
      if(bar_time < pred.start_time) continue;
      bool is_hit = (pred.direction == DIR_BULLISH && pred.target_price <= bar_high) || (pred.direction == DIR_BEARISH && pred.target_price >= bar_low);
      if(is_hit || bar_time >= pred.end_time)
        {
         g_total_predictions[pred.step]++;
         if(is_hit) g_total_hits[pred.step]++;
         ArrayRemove(g_pending_predictions, i, 1);
        }
     }
  }

void EnsureDataFolderExists() { if(!FolderCreate(DATA_FOLDER)) { PrintFormat("Warning: Could not create folder '%s'.", DATA_FOLDER); } }

void ManageTrailingStop()
  {
   if(!g_EnableTrailingStop || !PositionSelect(_Symbol)) return;
   double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN), currentSL = PositionGetDouble(POSITION_SL);
   long positionType = PositionGetInteger(POSITION_TYPE);
   MqlTick tick; if(!SymbolInfoTick(_Symbol, tick)) return;
   double pips_to_points = _Point * pow(10, _Digits % 2);
   if(positionType == POSITION_TYPE_BUY)
     {
      if((tick.bid - entryPrice) > (g_TrailingStartPips * pips_to_points))
        {
         double newSL = tick.bid - (g_TrailingStopPips * pips_to_points);
         if(newSL > currentSL || currentSL == 0) trade.PositionModify(_Symbol, newSL, PositionGetDouble(POSITION_TP));
        }
     }
   else if(positionType == POSITION_TYPE_SELL)
     {
      if((entryPrice - tick.ask) > (g_TrailingStartPips * pips_to_points))
        {
         double newSL = tick.ask + (g_TrailingStopPips * pips_to_points);
         if(newSL < currentSL || currentSL == 0) trade.PositionModify(_Symbol, newSL, PositionGetDouble(POSITION_TP));
        }
     }
  }
  
//+------------------------------------------------------------------+
//| MQL5 Main Event Handlers
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("=== Siobhan EA v1.62 (Confidence Filter) Initializing ===");
   InitializeParameters();
   if(MQLInfoInteger(MQL_TESTER)) { if(!LoadBacktestPredictions()) return(INIT_FAILED); }
   EnsureDataFolderExists();
   string symbols[]={Symbol_EURJPY,Symbol_USDJPY,Symbol_GBPUSD,Symbol_EURGBP,Symbol_USDCAD,Symbol_USDCHF};
   for(int i=0;i<ArraySize(symbols);i++) SymbolSelect(symbols[i],true);
   atr_handle=iATR(_Symbol,PERIOD_H1,g_ATR_Period); macd_handle=iMACD(_Symbol,PERIOD_H1,12,26,9,PRICE_CLOSE);
   rsi_handle=iRSI(_Symbol,PERIOD_H1,14,PRICE_CLOSE); stoch_handle=iStochastic(_Symbol,PERIOD_H1,14,3,3,MODE_SMA,STO_LOWHIGH);
   cci_handle=iCCI(_Symbol,PERIOD_H1,20,PRICE_TYPICAL); adx_handle=iADX(_Symbol,PERIOD_H1,g_ADX_Period);
   if(atr_handle==INVALID_HANDLE||macd_handle==INVALID_HANDLE||rsi_handle==INVALID_HANDLE||stoch_handle==INVALID_HANDLE||cci_handle==INVALID_HANDLE||adx_handle==INVALID_HANDLE)
     { Print("Error: Failed to create one or more indicator handles."); return(INIT_FAILED); }
   ArrayInitialize(g_last_predictions,0.0); ArrayInitialize(g_accuracy_pct,0.0);
   ArrayInitialize(g_total_hits,0); ArrayInitialize(g_total_predictions,0);
   ArrayFree(g_pending_predictions);
   CreateDisplayPanel();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   IndicatorRelease(atr_handle); IndicatorRelease(macd_handle); IndicatorRelease(rsi_handle);
   IndicatorRelease(stoch_handle); IndicatorRelease(cci_handle); IndicatorRelease(adx_handle);
   DeleteDisplayPanel(); Comment("");
  }

void OnTick()
  {
   if(PositionsTotal()>0&&PositionSelect(_Symbol))
     {
      MqlTick tick; if(SymbolInfoTick(_Symbol,tick))
        {
         if(UseMarketOrderForTP&&g_active_trade_target_price>0)
           {
            if((PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY&&tick.bid>=g_active_trade_target_price)||(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL&&tick.ask<=g_active_trade_target_price))
              { PrintFormat("Closing by market TP hit: #%d",(long)PositionGetInteger(POSITION_TICKET)); trade.PositionClose(_Symbol); g_active_trade_target_price=0; return; }
           }
         if(g_EnableTimeBasedExit)
           {
            datetime deadline=(datetime)PositionGetInteger(POSITION_TIME)+(g_MaxPositionHoldBars*PeriodSeconds(PERIOD_H1));
            MqlDateTime now; TimeToStruct(TimeCurrent(),now);
            if(TimeCurrent()>=deadline&&now.min>=g_ExitBarMinute)
              { PrintFormat("Closing by time limit: #%d",(long)PositionGetInteger(POSITION_TICKET)); trade.PositionClose(_Symbol); g_active_trade_target_price=0; return; }
           }
        }
      ManageTrailingStop();
      return;
     }

   static datetime last_bar_time=0;
   datetime current_bar_time=iTime(_Symbol,PERIOD_H1,0);
   if(current_bar_time==last_bar_time) return;
   last_bar_time=current_bar_time;

   g_active_trade_target_price=0;
   CheckPastPredictionAccuracy();
   if(TradingLogicMode==MODE_TRADING_DISABLED) return;
   
   double predicted_prices[PREDICTION_STEPS]; bool got_prediction=false;
   ArrayInitialize(predicted_prices,0.0);
   
   if(MQLInfoInteger(MQL_TESTER))
     {
      BacktestPrediction current_pred;
      if(FindPredictionForBar(iTime(_Symbol,PERIOD_H1,1),current_pred))
        {
         for(int i=0;i<PREDICTION_STEPS;i++){predicted_prices[i]=current_pred.predicted_prices[i];g_last_predictions[i]=predicted_prices[i];}
         got_prediction=true;
        }
     }
   else
     {
      double features[SEQ_LEN*FEATURE_COUNT]; int data_needed=SEQ_LEN+30;
      MqlRates rates[];if(CopyRates(_Symbol,PERIOD_H1,1,data_needed,rates)<data_needed)return;
      double macd[],rsi[],stoch_k[],cci[],ej_c[],uj_c[],gu_c[],eg_c[],uc_c[],uchf_c[];
      if(CopyBuffer(macd_handle,0,1,data_needed,macd)<data_needed||CopyBuffer(rsi_handle,0,1,data_needed,rsi)<data_needed||CopyBuffer(stoch_handle,0,1,data_needed,stoch_k)<data_needed||CopyBuffer(cci_handle,0,1,data_needed,cci)<data_needed)return;
      if(CopyClose(Symbol_EURJPY,PERIOD_H1,1,data_needed,ej_c)<data_needed||CopyClose(Symbol_USDJPY,PERIOD_H1,1,data_needed,uj_c)<data_needed||CopyClose(Symbol_GBPUSD,PERIOD_H1,1,data_needed,gu_c)<data_needed||CopyClose(Symbol_EURGBP,PERIOD_H1,1,data_needed,eg_c)<data_needed||CopyClose(Symbol_USDCAD,PERIOD_H1,1,data_needed,uc_c)<data_needed||CopyClose(Symbol_USDCHF,PERIOD_H1,1,data_needed,uchf_c)<data_needed)return;
      int feature_index=0;
      for(int i=SEQ_LEN-1;i>=0;i--)
        {
         double eurusd_ret=(rates[i].close/rates[i+1].close)-1.0,eurjpy_ret=(ej_c[i]/ej_c[i+1])-1.0,usdjpy_ret=(uj_c[i]/uj_c[i+1])-1.0,gbpusd_ret=(gu_c[i]/gu_c[i+1])-1.0,eurgbp_ret=(eg_c[i]/eg_c[i+1])-1.0,usdcad_ret=(uc_c[i]/uc_c[i+1])-1.0,usdchf_ret=(uchf_c[i]/uchf_c[i+1])-1.0;
         MqlDateTime dt;TimeToStruct(rates[i].time,dt);
         features[feature_index++]=eurusd_ret;features[feature_index++]=(double)rates[i].tick_volume;features[feature_index++]=(rates[i].high-rates[i].low);
         features[feature_index++]=macd[i];features[feature_index++]=rsi[i];features[feature_index++]=stoch_k[i];features[feature_index++]=cci[i];
         features[feature_index++]=(double)dt.hour;features[feature_index++]=(double)dt.day_of_week;
         features[feature_index++]=(usdjpy_ret+usdcad_ret+usdchf_ret)-(eurusd_ret+gbpusd_ret);
         features[feature_index++]=eurusd_ret+eurjpy_ret+eurgbp_ret;
         features[feature_index++]=-(eurjpy_ret+usdjpy_ret);
        }
      if(SendToDaemon("predict_regression",features,predicted_prices))
        {
         for(int i=0;i<PREDICTION_STEPS;i++)g_last_predictions[i]=predicted_prices[i];
         got_prediction=true;
        }
     }
     
   UpdateDisplayPanel();
   if(!got_prediction)return;

   if(!MQLInfoInteger(MQL_TESTER)&&predicted_prices[0]>0)
     {
      int bullish_steps=0,bearish_steps=0;MqlTick temp_tick;SymbolInfoTick(_Symbol,temp_tick);
      for(int i=0;i<PREDICTION_STEPS;i++){if(predicted_prices[i]>temp_tick.ask)bullish_steps++;if(predicted_prices[i]<temp_tick.bid)bearish_steps++;}
      if(bullish_steps>bearish_steps||bearish_steps>bullish_steps)
        {
         ENUM_PREDICTION_DIRECTION dir=(bullish_steps>bearish_steps)?DIR_BULLISH:DIR_BEARISH;
         for(int step=0;step<PREDICTION_STEPS;step++)
           {
            PendingPrediction pred;pred.target_price=predicted_prices[step];pred.start_time=TimeCurrent();pred.end_time=TimeCurrent()+(AccuracyLookaheadBars*PeriodSeconds(PERIOD_H1));
            pred.direction=dir;pred.step=step;int sz=ArraySize(g_pending_predictions);ArrayResize(g_pending_predictions,sz+1);g_pending_predictions[sz]=pred;
           }
        }
     }

   if(g_EnableADXFilter){double adx[];if(CopyBuffer(adx_handle,0,1,1,adx)<1||adx[0]<g_ADX_Threshold)return;}
   MqlTick latest_tick;if(!SymbolInfoTick(_Symbol,latest_tick))return;
   double atr_val[];if(CopyBuffer(atr_handle,0,1,1,atr_val)<1)return;
   double pips_to_points=_Point*pow(10,_Digits%2);

   if(TradingLogicMode==MODE_REGRESSION_ONLY||TradingLogicMode==MODE_COMBINED)
     {
      double target_price=predicted_prices[TakeProfitTargetBar];
      double spread_points=SymbolInfoInteger(_Symbol,SYMBOL_SPREAD)*_Point;
      double min_profit_points=g_MinProfitPips*pips_to_points;
      int bullish_steps=0,bearish_steps=0;
      for(int i=0;i<PREDICTION_STEPS;i++){if(predicted_prices[i]>latest_tick.ask)bullish_steps++;if(predicted_prices[i]<latest_tick.bid)bearish_steps++;}

      double bullish_confidence = (double)bullish_steps / PREDICTION_STEPS;
      double bearish_confidence = (double)bearish_steps / PREDICTION_STEPS;

      if(bullish_steps>=g_RequiredConsistentSteps&&bullish_confidence>=g_MinimumConfidence&&(target_price-latest_tick.ask)>(min_profit_points+spread_points))
        {
         double sl=(StopLossMode==SL_STATIC_PIPS)?latest_tick.ask-(g_StaticStopLossPips*pips_to_points):latest_tick.ask-(atr_val[0]*g_ATR_SL_Multiplier);
         if(latest_tick.ask-sl>0&&(target_price-latest_tick.ask)/(latest_tick.ask-sl)>=g_MinimumRiskRewardRatio)
           {
            double tp;
            if(TakeProfitMode==TP_REGRESSION_TARGET)tp=target_price;
            else if(TakeProfitMode==TP_STATIC_PIPS)tp=latest_tick.ask+(g_StaticTakeProfitPips*pips_to_points);
            else tp=latest_tick.ask+(atr_val[0]*g_ATR_TP_Multiplier);
            if(UseMarketOrderForTP){tp=0;g_active_trade_target_price=(TakeProfitMode==TP_REGRESSION_TARGET)?target_price:tp;}
            double lots=CalculateLotSize(sl,latest_tick.ask);
            if(lots>0&&trade.Buy(lots,_Symbol,latest_tick.ask,sl,tp,"LSTM Regress Buy"))return;
           }
        }
      else if(bearish_steps>=g_RequiredConsistentSteps&&bearish_confidence>=g_MinimumConfidence&&(latest_tick.bid-target_price)>(min_profit_points+spread_points))
        {
         double sl=(StopLossMode==SL_STATIC_PIPS)?latest_tick.bid+(g_StaticStopLossPips*pips_to_points):latest_tick.bid+(atr_val[0]*g_ATR_SL_Multiplier);
         if(sl-latest_tick.bid>0&&(latest_tick.bid-target_price)/(sl-latest_tick.bid)>=g_MinimumRiskRewardRatio)
           {
            double tp;
            if(TakeProfitMode==TP_REGRESSION_TARGET)tp=target_price;
            else if(TakeProfitMode==TP_STATIC_PIPS)tp=latest_tick.bid-(g_StaticTakeProfitPips*pips_to_points);
            else tp=latest_tick.bid-(atr_val[0]*g_ATR_TP_Multiplier);
            if(UseMarketOrderForTP){tp=0;g_active_trade_target_price=(TakeProfitMode==TP_REGRESSION_TARGET)?target_price:tp;}
            double lots=CalculateLotSize(sl,latest_tick.bid);
            if(lots>0&&trade.Sell(lots,_Symbol,latest_tick.bid,sl,tp,"LSTM Regress Sell"))return;
           }
        }
     }
  }

double OnTester()
  {
   double history_profits[]; HistorySelect(0, TimeCurrent());
   int deals = HistoryDealsTotal(), profit_count = 0; if(deals <= 1) return 0.0;
   ArrayResize(history_profits, deals);
   for(int i = 0; i < deals; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket > 0 && HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) history_profits[profit_count++] = HistoryDealGetDouble(ticket, DEAL_PROFIT);
     }
   if(profit_count <= 1) return 0.0; ArrayResize(history_profits, profit_count);
   double mean_profit = MathMean(history_profits);
   double std_dev_profit = MathStandardDeviation(history_profits);
   if(std_dev_profit < 0.0001) return 0.0;
   double sharpe_ratio = mean_profit / std_dev_profit;
   double custom_criterion = sharpe_ratio * MathSqrt(profit_count);
   PrintFormat("OnTester Pass Complete: Trades=%d, Mean Profit=%.2f, StdDev=%.2f, Sharpe=%.3f, Custom Criterion=%.3f", profit_count, mean_profit, std_dev_profit, sharpe_ratio, custom_criterion);
   return custom_criterion;
  }
//+------------------------------------------------------------------+
