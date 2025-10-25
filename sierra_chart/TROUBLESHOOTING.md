# Sierra Chart DLL Troubleshooting

## Issue: DLL Not Showing in Studies List

### Possible Causes:

1. **Missing Dependencies** - DLL requires gRPC runtime DLLs
2. **Sierra Chart Not Restarted** - Need to restart after adding DLL
3. **Wrong Architecture** - DLL must be 64-bit if Sierra Chart is 64-bit
4. **Export Issue** - Function not properly exported

---

## Solution 1: Copy gRPC DLLs to Sierra Chart

The DLL needs gRPC runtime libraries. Copy them:

```powershell
# Copy all required DLLs
copy C:\vcpkg\installed\x64-windows\bin\*.dll C:\SierraChart\
```

---

## Solution 2: Restart Sierra Chart

1. **Close Sierra Chart completely**
2. Wait 5 seconds
3. **Restart Sierra Chart**
4. Try adding the study again

---

## Solution 3: Check DLL Architecture

```powershell
# Check if DLL is 64-bit
dumpbin /headers C:\SierraChart\Data\NexusAI.dll | findstr machine
```

Should show: `x64` or `8664 machine (x64)`

---

## Solution 4: Create Simple Test DLL

Create a minimal DLL without gRPC to test if Sierra Chart can load it:

```cpp
#include "sierrachart.h"

SCDLLName("NEXUS AI Test")

SCSFExport scsf_NexusAI_Test(SCStudyInterfaceRef sc) {
    if (sc.SetDefaults) {
        sc.GraphName = "NEXUS AI Test";
        sc.StudyDescription = "Simple test - no gRPC";
        sc.AutoLoop = 1;
        return;
    }
    
    if (sc.Index % 100 == 0) {
        sc.AddMessageToLog("NEXUS AI Test is working!", 0);
    }
}
```

---

## Most Likely Fix: Copy Runtime DLLs

Run this command:

```powershell
cd C:\Users\Nexus AI\Documents\NEXUS\sierra_chart
.\copy_dlls.bat
```

This will copy all gRPC runtime DLLs to Sierra Chart folder.

---

## Check Sierra Chart Message Log

1. Open Sierra Chart
2. Global Settings → Message Log
3. Look for errors about NexusAI.dll
4. Common errors:
   - "Cannot load DLL" → Missing dependencies
   - "Entry point not found" → Export issue
   - No error → DLL not in correct folder
