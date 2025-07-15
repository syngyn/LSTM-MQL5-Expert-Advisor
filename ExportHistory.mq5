//+------------------------------------------------------------------+
//|                                               ExportHistory.mq5  |
//|                    Exports H1 data for all required symbols      |
//|                                           Jason.w.rusk@gmail.com |
//+------------------------------------------------------------------+
#property copyright "Jason Rusk 2025"
#property version   "1.00"
#property script_show_inputs

#include <Files/File.mqh>

// --- Inputs: List of symbols to export ---
input string Symbol1 = "EURUSD";
input string Symbol2 = "EURJPY";
input string Symbol3 = "USDJPY";
input string Symbol4 = "GBPUSD";
input string Symbol5 = "EURGBP";
input string Symbol6 = "USDCAD";
input string Symbol7 = "USDCHF";

// --- Input: How many bars to export ---
input int BarsToExport = 50000; // Export a large number to ensure full history

//+------------------------------------------------------------------+
//| Export data for a single symbol
//+------------------------------------------------------------------+
bool ExportSymbol(string symbol)
{
   PrintFormat("Exporting %s...", symbol);

   // Ensure the symbol is available in Market Watch
   if(!SymbolSelect(symbol, true))
   {
      PrintFormat("Could not select symbol %s. Please add it to Market Watch.", symbol);
      return false;
   }
   
   // Copy the historical data
   MqlRates rates[];
   int bars_copied = CopyRates(symbol, PERIOD_H1, 0, BarsToExport, rates);
   if(bars_copied <= 0)
   {
      PrintFormat("Failed to copy rates for %s. Error: %d", symbol, GetLastError());
      return false;
   }

   // Open the file for writing. This will place it in the MQL5/Files directory
   // which is where our Python scripts can access it.
   string filename = symbol + "60.csv";
   int file_handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(file_handle == INVALID_HANDLE)
   {
      PrintFormat("Failed to open file %s for writing. Error: %d", filename, GetLastError());
      return false;
   }

   // Write header
   FileWrite(file_handle, "date", "time", "open", "high", "low", "close", "tickvol");

   // Write the data
   for(int i = 0; i < bars_copied; i++)
   {
      MqlDateTime dt;
      TimeToStruct(rates[i].time, dt);
      
      string date_str = StringFormat("%04u.%02u.%02u", dt.year, dt.mon, dt.day);
      string time_str = StringFormat("%02u:%02u", dt.hour, dt.min);

      FileWrite(file_handle, date_str, time_str, rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume);
   }

   FileClose(file_handle);
   PrintFormat("Successfully exported %d bars to %s.", bars_copied, filename);
   return true;
}

//+------------------------------------------------------------------+
//| Script program start function
//+------------------------------------------------------------------+
void OnStart()
{
   Print("--- Starting Automated History Export ---");
   
   string symbols_to_export[] = {Symbol1, Symbol2, Symbol3, Symbol4, Symbol5, Symbol6, Symbol7};
   int success_count = 0;

   for(int i = 0; i < ArraySize(symbols_to_export); i++)
   {
      if(ExportSymbol(symbols_to_export[i]))
      {
         success_count++;
      }
   }
   
   PrintFormat("--- Export Complete. Successfully exported %d of %d symbols. ---", success_count, ArraySize(symbols_to_export));
}