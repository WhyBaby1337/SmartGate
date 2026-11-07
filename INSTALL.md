# Smart Gate — Windows VST3 installation

## Requirements

- Windows x64.
- A DAW that supports VST3 plug-ins.
- Microsoft Visual C++ 2015–2022 Redistributable (x64). Install it if it is not already present on the system.

The ONNX Runtime DLLs and the Silero VAD model are included with the plug-in bundle. Do not download or copy separate ONNX Runtime files into the VST3 folder.

## Install

1. Close your DAW.
2. Download the `SmartGate-Windows-x64-VST3` artifact ZIP from the successful Windows workflow run.
3. Extract the archive. It contains the complete `Smart Gate.vst3` bundle.
4. Copy the entire `Smart Gate.vst3` folder to the system VST3 directory:

   `%CommonProgramFiles%\VST3`

   Usually this resolves to `C:\Program Files\Common Files\VST3`. Create the `VST3` directory if it does not exist. Windows may ask for administrator permission.

5. Start your DAW and run its VST3 plug-in rescan if Smart Gate does not appear automatically.

Do not copy only the file named `Smart Gate.vst3` from inside the bundle. Keep the full folder structure, including `Contents\x86_64-win` and `Contents\Resources`, so the bundled runtime dependencies and notices remain beside the module.

## Troubleshooting

- **The plug-in is missing from the DAW:** confirm that the full `Smart Gate.vst3` folder is under the system VST3 directory, then rescan VST3 plug-ins.
- **Windows reports a missing runtime DLL:** install or repair the official Microsoft Visual C++ 2015–2022 Redistributable for x64, then restart the DAW.
- **The plug-in still fails to load:** check the DAW's plug-in scan or validation log. Do not replace the bundled `onnxruntime.dll` or `onnxruntime_providers_shared.dll` with files from another ONNX Runtime version.

## CPU meter

The editor's **PLUGIN CPU** readout estimates Smart Gate's own `processBlock` time as a percentage of the real-time budget for the current audio block. `100%` means the plug-in used the entire block deadline; values above `100%` indicate that processing exceeded that block's budget. The smoothed value is shown alongside a short peak hold. This is not the computer's total CPU usage and does not include other plug-ins or DAW processing.

## Build validation status

The CI workflow builds the Windows x64 VST3, runs DSP and processor-state tests, runs Silero smoke/latency tests, checks the bundle dependencies, and validates the module with pluginval. The CI pluginval run skips GUI tests because the runner is headless. A successful CI run does not replace testing the plug-in in the target DAW on the user's computer.
