@echo off
setlocal enabledelayedexpansion
title Qwen3.6/Qwen3.8/KAT llama-server launcher
color 0B

:: Re-entered elevated to run the GPU reset (option 6)
if /i "%~1"=="gpu_reset" goto gpu_reset_elevated

set "MODEL_IQ=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
set "MODEL_KATXS=C:/Users/BhatSahab/.lmstudio/models/thread13/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF-MTP/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.bartowski.mtp.gguf"
set "MODEL_Q38=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf"
set "MODEL_Q38M=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_M.gguf"
set "MODEL_Q3X=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf"
set "MODEL_Q2X=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf"
set "MODEL_RIDGE=C:/Users/BhatSahab/.lmstudio/models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf"
set "MODEL_Q8=C:/Users/BhatSahab/.lmstudio/models/empero-ai/Qwen3.8-9B-Distill-GGUF/Qwen3.8-9B-Q8_0.gguf"
:: DFlash2 draft (Qwen3.8-27B family, draft-dflash) - same relative path as run_model.sh
set "DFLASH_DRAFT=C:/Users/BhatSahab/.lmstudio/models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
set "SAMPLING_FLAG=--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0"
set "Q38_FLAG="
set "KV_K=q4_0" & set "KV_V=q4_0"
set "CACHE_RAM=0"
set "VULKAN_DIR=E:\Work\llama\backends\bin\vulkan-b10181"
set "ROCM_DIR=E:\Work\llama\backends\bin\rocm-selfbuilt"
set "Q38_DIR=E:\Work\llama\backends\bin\vulkan"
set "STEW_DIR=E:\Work\llama\backends\bin\rocm-stew675"
set "VPATCH_DIR=E:\Work\llama\backends\bin\vulkan-patched"
set "ROCM72_DIR=E:\Work\llama\backends\bin\rocm-72"
set "ROCMOF_DIR=E:\Work\llama\backends\bin\rocm"
set "PERF_DIR=E:\Work\llama\backends\bin\rocm-perf"
set "ZUID_DIR=E:\Work\llama\backends\bin\vulkan-zuid"

:: ── Model selection ────────────────────────────────
:menu_model
set "KV_K=q4_0" & set "KV_V=q4_0"
set "CACHE_RAM=0"
set "SPEC_FLAG="
set "MTP_MODE=off"
set "DFLASH_MODE=off"
set "MOE_CACHE_PROFILE="
set "MOE_CACHE_SLOTS=0"
set "MOE_CACHE_FLAG="
set "MMPROJ_PATH="
set "MMPROJ_FLAG="
:: rocm-perf backend sets these; clear so other backends don't inherit
set "GGML_CUDA_REGISTER_HOST="
set "GGML_SCHED_PREFETCH_EXPERTS="
cls
echo ================================================
echo   Qwen3.6 / KAT-Coder / Qwen3.8 - Launcher
echo ================================================
echo.
echo  Select model:
echo   [1] Qwen3.6 IQ4_XS MTP (18.2 GB, MTP, tg ~37 / 49 w/ MTP, vision)
echo   [2] KAT-Coder IQ4_XS MTP (19.7 GB, tg ~53)
echo   [3] Qwen3.8-27B IQ4_XS (14.3 GB, dense, tg ~7-15 ⚠️ regression, vision)
echo   [4] Qwen3.8-27B Q3_K_XL (13.1 GB, dense, tg ~33 / MTP 38, vision)
echo   [5] Qwen3.8-27B Q3_K_M (13.8 GB, dense, tg ~37, vision)
echo   [6] Qwen3.8-27B Ridge-3.7bpw (12.6 GB, dense, ALL ctx ✅, vision)
echo   [7] Qwen3.8-27B Q2_K_XL (~9 GB, tg ~37 / MTP 45, vision)
echo   [8] Qwen3.8-9B-Distill Q8_0 (9.1 GB, q8_0 KV, tg ~58)
echo   [9] Reset GPU (pnputil restart - fixes stuck idle clocks)
echo   [0] GPU Power Profile (ADLX: low/stock/max)
echo.
choice /c 1234567890 /n /m "  Choice [1-9,0]: "
if errorlevel 10 goto menu_power
if errorlevel 9 goto gpu_reset
if errorlevel 8 goto model_q8
if errorlevel 7 goto model_q2x
if errorlevel 6 goto model_ridge
if errorlevel 5 goto model_q38m
if errorlevel 4 goto model_q3x
if errorlevel 3 goto model_q38
if errorlevel 2 goto model_katxs
set "MODEL=%MODEL_IQ%"
set "MODEL_IS_MTP=1"
set "MMPROJ_PATH=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/mmproj-F32.gguf"
goto menu_backend

:model_katxs
set "MODEL=%MODEL_KATXS%"
set "MODEL_IS_MTP=1"
goto menu_backend

:model_q38
set "MODEL=%MODEL_Q38%"
set "MODEL_IS_MTP=1"
set "Q38_FLAG=--reasoning on --chat-template-kwargs {^"reasoning_effort^":^"medium^",^"preserve_reasoning^":true}"
set "MMPROJ_PATH=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf"
goto menu_backend_dense

:model_q38m
set "MODEL=%MODEL_Q38M%"
set "MODEL_IS_MTP=1"
set "Q38_FLAG=--reasoning on --chat-template-kwargs {^"reasoning_effort^":^"medium^",^"preserve_reasoning^":true}"
set "MMPROJ_PATH=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf"
goto menu_backend_dense

:model_q3x
set "MODEL=%MODEL_Q3X%"
set "MODEL_IS_MTP=1"
:: Q3_K_XL 13.1 GB - only vulkan tested (tg 32.8 no-MTP / 38.3 MTP n_max=1 @32K). Same config as Q2_K_XL.
set "Q38_FLAG=--reasoning on --chat-template-kwargs {^"reasoning_effort^":^"medium^",^"preserve_reasoning^":true}"
set "MMPROJ_PATH=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf"
goto menu_backend_dense

:model_q2x
set "MODEL=%MODEL_Q2X%"
set "MODEL_IS_MTP=1"
:: Q2_K_XL ~9 GB - fastest Q38 family tg (34 no-MTP / 53.4 MTP n_max=1 @32K, 2026-08-22). stew no-MTP crashes (MUL_MAT).
set "Q38_FLAG=--reasoning on --chat-template-kwargs {^"reasoning_effort^":^"medium^",^"preserve_reasoning^":true}"
set "MMPROJ_PATH=C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf"
goto menu_backend_dense

:model_q8
set "MODEL=%MODEL_Q8%"
set "MODEL_IS_MTP=1"
:: Qwen3.8-9B-Distill - small hybrid, thinking flags. Q8_0 weights + q8_0 KV (tg 58.2). MTP selfbuilt 94 (Q4_K_M) - Q8 MTP pending test.
set "Q38_FLAG=--reasoning on --chat-template-kwargs {^"reasoning_effort^":^"medium^",^"preserve_reasoning^":true}"
set "KV_K=q8_0" & set "KV_V=q8_0"
:: no 9B-compatible mmproj on disk (27B ones die - hidden-dim mismatch)
goto menu_backend_dense

:model_ridge
set "MODEL=%MODEL_RIDGE%"
set "MODEL_IS_MTP=1"
:: Ridge 3.7bpw dense (empero) - best dense: ALL ctx incl 256K.
set "Q38_FLAG=--reasoning on --chat-template-kwargs {^"reasoning_effort^":^"medium^",^"preserve_reasoning^":true}"
set "MMPROJ_PATH=C:/Users/BhatSahab/.lmstudio/models/empero-ai/Qwen3.8-27B-Ridge-GGUF/mmproj-Qwen3.8-27B-BF16.gguf"
goto menu_backend_dense

:: ── GPU power profile (ADLX, live, resets on reboot) ──
:menu_power
cls
echo ================================================
echo   GPU Power Profile - RX 9070 XT (ADLX)
echo ================================================
echo.
echo  Live tuning, reverts on reboot:
echo   [1] Low power  (-30%%, quiet, pp -14%%)
echo   [2] Stock
echo   [3] Max power (+10%%, no gain, not power-limited)
echo.
choice /c 123 /n /m "  Profile [1-3]: "
if errorlevel 3 goto power_max
if errorlevel 2 goto power_stock
"%~dp0tools\adlx-profile.exe" apply --power-limit -30
goto power_done

:power_stock
"%~dp0tools\adlx-profile.exe" reset
goto power_done

:power_max
"%~dp0tools\adlx-profile.exe" apply --power-limit 10
goto power_done

:power_done
echo.
echo  Done. Returning to main menu...
timeout /t 2 /nobreak >nul
goto menu_model

:: ── GPU reset (pnputil /restart-device, fallback PnP disable/enable) ──
:gpu_reset
cls
echo ================================================
echo   GPU Reset - RX 9070 XT (pnputil restart)
echo ================================================
echo.
echo   Screen will go black for a few seconds.
echo   Fixes GPU stuck at idle clocks (tg collapse).
echo   Requires admin - UAC prompt if not elevated.
echo.
net session >nul 2>&1
if errorlevel 1 (
  echo  Elevating to admin...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList 'gpu_reset' -Verb RunAs -Wait"
  echo.
  echo  Reset done. Return to menu...
  timeout /t 2 /nobreak >nul
  goto menu_model
)
:gpu_reset_elevated
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\reset-gpu.ps1"
echo.
if /i "%~1"=="gpu_reset" (
  echo  GPU reset finished. Closing elevated window...
  timeout /t 3 /nobreak >nul
  exit
)
echo  Return to menu...
timeout /t 2 /nobreak >nul
goto menu_model

:: ── Backend selection ──────────────────────────────
:menu_backend
set "MOE_BACKEND=0"
cls
echo ================================================
echo  Model: %MODEL%
echo ================================================
echo.
echo  Select backend:
echo   [1] Vulkan b10181 (tg ~46 / MTP 53)
echo   [2] ROCm selfbuilt (b10488+fix+zuid, tg ~36 / MTP 53.6)
echo   [3] ROCm stew675 (b10486, tg ~35 / MTP 41)
echo   [4] Vulkan patched (b10488+pi, tg ~32 / MTP 39)
echo   [5] ROCm 7.2 selfbuilt (tg ~35 / MTP 45)
echo   [6] ROCm official b10612 + 7.14 runtime (tg ~35 / MTP 46)
echo   [7] ROCm perf (Fable fork, pp +52%% @ ncmoe26, MTP ~48)
echo   [8] Vulkan zuid b10641 (fork: pp 1480 / MTP tg 60 on Qwen3.6 🏆)
echo.
choice /c 12345678 /n /m "  Backend [1-8]: "
if errorlevel 8 goto backend_moe_zuid
if errorlevel 7 goto backend_moe_perf
if errorlevel 6 goto backend_moe_rocmof
if errorlevel 5 goto backend_moe_rocm72
if errorlevel 4 goto backend_moe_vpatch
if errorlevel 3 goto backend_moe_stew
if errorlevel 2 goto backend_rocm
set "SERVER=%VULKAN_DIR%\llama-server.exe"
set "BACKEND_IS_VK=1"
set "BACKEND_NAME=vulkan"
goto menu_vision

:backend_moe_vpatch
set "SERVER=%VPATCH_DIR%\llama-server.exe"
set "BACKEND_IS_VK=1"
set "BACKEND_NAME=vulkan-patched"
goto menu_vision

:backend_moe_rocm72
set "SERVER=%ROCM72_DIR%\llama-server.exe"
set "BACKEND_IS_VK=0"
set "BACKEND_NAME=rocm-72"
goto menu_vision

:backend_moe_rocmof
set "SERVER=%ROCMOF_DIR%\llama-server.exe"
set "BACKEND_IS_VK=0"
set "BACKEND_NAME=rocm-official"
goto menu_vision

:backend_moe_perf
set "SERVER=%PERF_DIR%\llama-server.exe"
set "BACKEND_IS_VK=0"
set "BACKEND_NAME=rocm-perf"
set "MOE_BACKEND=1"
:: Fable fork pp optimizations (DMA pinning + expert prefetch) - pp +52%% @ high ncmoe
set "GGML_CUDA_REGISTER_HOST=1"
set "GGML_SCHED_PREFETCH_EXPERTS=1"
goto menu_vision

:backend_moe_zuid
set "SERVER=%ZUID_DIR%\llama-server.exe"
set "BACKEND_IS_VK=1"
set "BACKEND_NAME=vulkan-zuid"
:: LaurentZuijdwijk fork b10641 - LDS bank-conflict pad + f16 B operand + GDN concat-transpose + adaptive spec.
:: Qwen3.6 MTP n=3 warm: pp 1480 / tg 59.7 (b10612: 440/56.2). Q38 MTP n=1: tie with b10612.
:: Q38 adaptive spec NOT usable (n_min>=3 collapses). Keep plain draft-mtp for Qwen3.6 too.
goto menu_vision

:backend_rocm
set "SERVER=%ROCM_DIR%\llama-server.exe"
set "BACKEND_IS_VK=0"
set "BACKEND_NAME=selfbuilt"
goto menu_vision

:backend_moe_stew
set "SERVER=%STEW_DIR%\llama-server.exe"
set "BACKEND_IS_VK=0"
set "BACKEND_NAME=stew"
goto menu_vision

:: ── Dense backend selection (Q38/Ridge) ────────────
:menu_backend_dense
set "MOE_BACKEND=0"
cls
echo ================================================
echo  Model: %MODEL%
echo ================================================
echo.
echo  Select backend:
echo   [1] Vulkan  (b10488, tg ~31-37)
echo   [2] ROCm stew675  (b10486, pp ~1000, tg ~30)
echo   [3] ROCm selfbuilt (b10488+fix, pp ~900, MTP works)
echo   [4] ROCm official b10488 + 7.2 runtime
echo.
choice /c 1234 /n /m "  Backend [1-4]: "
if errorlevel 4 goto backend_dense_rocmof
if errorlevel 3 goto backend_dense_selfbuilt
if errorlevel 2 goto backend_dense_stew
set "SERVER=%Q38_DIR%\llama-server.exe"
set "BACKEND_IS_VK=1"
set "BACKEND_NAME=vulkan"
goto menu_vision

:backend_dense_stew
set "SERVER=%STEW_DIR%\llama-server.exe"
set "BACKEND_IS_VK=0"
set "BACKEND_NAME=stew"
goto menu_vision

:backend_dense_selfbuilt
set "SERVER=%ROCM_DIR%\llama-server.exe"
set "BACKEND_IS_VK=0"
set "BACKEND_NAME=selfbuilt"
goto menu_vision

:backend_dense_rocmof
set "SERVER=%ROCMOF_DIR%\llama-server.exe"
set "BACKEND_IS_VK=0"
set "BACKEND_NAME=rocm-official"
goto menu_vision

:: ── Vision toggle (models with mmproj on disk) ─────
:menu_vision
if "%MMPROJ_PATH%"=="" goto vision_off
:: stew fork (b10486) crashes on vision encoder (MUL_MAT) - gate it
if "%BACKEND_NAME%"=="stew" goto vision_off_stew
cls
echo ================================================
echo  Model  : %MODEL%
echo  Backend: %SERVER%
echo ================================================
echo.
echo  Vision (multimodal projector):
echo   [1] ON - CPU  (default: --no-mmproj-offload, saves ~0.9 GB VRAM)
echo   [2] ON - GPU  (faster image encode)
echo   [3] OFF
echo.
choice /c 123 /n /m "  Vision [1-3]: "
if errorlevel 3 goto vision_off
if errorlevel 2 goto vision_gpu
set "MMPROJ_FLAG=--mmproj %MMPROJ_PATH% --no-mmproj-offload"
goto menu_ctx

:vision_gpu
set "MMPROJ_FLAG=--mmproj %MMPROJ_PATH%"
goto menu_ctx

:vision_off_stew
cls
echo ================================================
echo  Vision NOT available on stew backend
echo  (b10486 fork: MUL_MAT crash on vision encoder)
echo  Continuing without vision...
echo ================================================
echo.
timeout /t 2 /nobreak >nul
set "MMPROJ_FLAG="
goto menu_ctx

:vision_off
set "MMPROJ_FLAG="
goto menu_ctx

:: ── Context selection ──────────────────────────────
:menu_ctx
cls
echo ================================================
echo  Model  : %MODEL%
echo  Backend: %SERVER%
echo ================================================
echo.
call :set_speeds
echo  Select context window (pp ^| tg t/s):
echo   [1] 32K   pp %PP32% ^| tg %SPD32%
echo   [2] 64K   pp %PP64% ^| tg %SPD64%
echo   [3] 80K   pp %PP80% ^| tg %SPD80%
echo   [4] 100K  pp %PP100% ^| tg %SPD100%
echo   [5] 128K  pp %PP128% ^| tg %SPD128%
echo   [6] 150K  pp %PP150% ^| tg %SPD150%
echo   [7] 180K  pp %PP180% ^| tg %SPD180%
echo   [8] 200K  pp %PP200% ^| tg %SPD200%
echo   [9] 256K  pp %PP256% ^| tg %SPD256%
echo.
choice /c 123456789 /n /m "  Context [1-9]: "
if errorlevel 9 goto ctx_256k
if errorlevel 8 goto ctx_200k
if errorlevel 7 goto ctx_180k
if errorlevel 6 goto ctx_150k
if errorlevel 5 goto ctx_128k
if errorlevel 4 goto ctx_100k
if errorlevel 3 goto ctx_80k
if errorlevel 2 goto ctx_64k
set "CTX_SIZE=32768"
goto menu_mtp

:ctx_80k
set "CTX_SIZE=81920"
goto menu_mtp

:ctx_100k
set "CTX_SIZE=102400"
goto menu_mtp

:ctx_128k
set "CTX_SIZE=131072"
goto menu_mtp

:ctx_150k
set "CTX_SIZE=153600"
goto menu_mtp

:ctx_180k
set "CTX_SIZE=184320"
goto menu_mtp

:ctx_200k
set "CTX_SIZE=204800"
goto menu_mtp

:ctx_256k
set "CTX_SIZE=262144"
goto menu_mtp

:: ── Expected pp/tg speeds per model (Vulkan, clean state) ──
:set_speeds
set "PP32=~2900" & set "PP64=~2500" & set "PP128=~2400" & set "PP180=~1200" & set "PP200=~1200" & set "PP256=~1200"
set "SPD32=~37" & set "SPD64=~48" & set "SPD128=~47" & set "SPD180=~40" & set "SPD200=~40" & set "SPD256=~45"
if "%MODEL%"=="%MODEL_IQ%" (
  :: Qwen3.6 MTP file - baseline tg 37, +31% w/ draft-mtp (rocm-selfbuilt). 64K+ = estimate.
  set "PP32=~2900" & set "PP64=~2500" & set "PP128=~2400" & set "PP180=~1200" & set "PP200=~1200" & set "PP256=~1200"
  set "SPD32=~37" & set "SPD64=~48" & set "SPD128=~47" & set "SPD180=~40" & set "SPD200=~40" & set "SPD256=~45"
)
:: IQ4_XS (13.3 GiB) - REGRESSION 2026-08-26: tg 7-15 on b10612 + selfbuilt (was 28.4 on b10488). pp 36 b10612 / 117 selfbuilt
if "%MODEL%"=="%MODEL_Q38%" (
  set "PP32=~40" & set "PP64=~40" & set "PP128=~40" & set "PP256=❌"
  set "SPD32=~7" & set "SPD64=~7" & set "SPD128=~7" & set "SPD256=❌"
)
if "%MODEL%"=="%MODEL_Q38M%" (
  :: Q3_K_M dense - 256K broken (VRAM compute buffers, pp ~100). KV ~36 KiB/tok.
  set "PP32=~655" & set "PP64=~646" & set "PP128=~570" & set "PP256=❌102"
  set "SPD32=~37" & set "SPD64=~32" & set "SPD128=~35" & set "SPD256=❌"
)
:: Q3_K_XL 13.1 GB - tg 32.8 no-MTP / 38.3 MTP @32K (vulkan). 256K = 13.1 + 4.7 KV over VRAM -> ❌ like Q3_K_M
if "%MODEL%"=="%MODEL_Q3X%" (
  set "PP32=~650" & set "PP64=~650" & set "PP128=~650" & set "PP256=❌"
  set "SPD32=~33" & set "SPD64=~33" & set "SPD128=~33" & set "SPD256=❌"
)
:: Q2_K_XL ~9 GB - MTP n=1 warm 2026-08-26: b10612 45.4 / zuid 45.1 / selfbuilt 43. no-MTP 37.1. pp ~680. 256K may fit (untested)
if "%MODEL%"=="%MODEL_Q2X%" (
  set "PP32=~650" & set "PP64=~650" & set "PP128=~650" & set "PP256=~?"
  set "SPD32=~45" & set "SPD64=~45" & set "SPD128=~45" & set "SPD256=~?"
)
if "%MODEL%"=="%MODEL_Q8%" (
  :: 9B Q8_0 + q8_0 KV - pp 2625-2713, tg 58.2 @32K. Small KV.
  set "PP32=~2670" & set "PP64=~2670" & set "PP128=~2670" & set "PP256=~2670"
  set "SPD32=~58" & set "SPD64=~58" & set "SPD128=~58" & set "SPD256=~58"
)
if "%MODEL%"=="%MODEL_RIDGE%" (
  :: best dense - ALL ctx work (pp 720-770, tg 27-35). KV ~36 KiB/tok fits RAM.
  set "PP32=~722" & set "PP64=~770" & set "PP128=~733" & set "PP256=~720"
  set "SPD32=~34" & set "SPD64=~35" & set "SPD128=~29" & set "SPD256=~27"
)
:: 80K/100K/150K aren't in the original speed tables - inherit 64K/128K by proximity
set "PP80=%PP64%" & set "SPD80=%SPD64%"
set "PP100=%PP128%" & set "SPD100=%SPD128%"
set "PP150=%PP128%" & set "SPD150=%SPD128%"
exit /b

:: ── MTP toggle (all models) ────────────────────────
:menu_mtp
:: MTP works everywhere EXCEPT: stew+qwen35moe (crash, q36/KAT).
:: DFlash2 (draft-dflash) on all Qwen3.8-27B models with a b10701+ build
set "DFLASH_OK=0"
set "_DVER="
if "%MODEL%"=="%MODEL_Q38%" set "DFLASH_OK=1"
if "%MODEL%"=="%MODEL_Q38M%" set "DFLASH_OK=1"
if "%MODEL%"=="%MODEL_Q3X%" set "DFLASH_OK=1"
if "%MODEL%"=="%MODEL_Q2X%" set "DFLASH_OK=1"
if "%MODEL%"=="%MODEL_RIDGE%" set "DFLASH_OK=1"
:: perf fork (Fable) doesn't offer draft-dflash - same as the rocm-linux/vulkan-linux gate in run_model.sh
if "%DFLASH_OK%"=="1" if "%MOE_BACKEND%"=="1" set "DFLASH_OK=0"
if not "%DFLASH_OK%"=="1" goto mtp_show
:: version banner is printed to stderr: "version: 0.3.0-dev (build 10701, commit ...)"
for /f "tokens=4 delims=(, " %%n in ('"%SERVER%" --version 2^>^&1 ^| findstr /c:"(build "') do set "_DVER=%%n"
if defined _DVER if %_DVER% GEQ 10701 set "DFLASH_OK=1"
:mtp_show
cls
echo ================================================
echo  Model  : %MODEL%
echo  Backend: %SERVER%
echo  Context: %CTX_SIZE%
echo ================================================
echo.
echo  MTP speculative decoding:
echo   [1] Normal MTP  (draft-mtp)
echo   [2] Tuned MTP   (per-backend: vk combined / stew adaptive / selfbuilt adaptive)
echo   [3] OFF         (safe)
if "%DFLASH_OK%"=="1" (
  echo   [4] DFlash2          ^(draft-dflash, KV4 draft, ~2x tg^)
  echo   [5] DFlash2 + ngram  ^(draft-dflash,ngram-mod on top^)
  choice /c 12345 /n /m "  MTP [1-5]: "
) else (
  choice /c 123 /n /m "  MTP [1-3]: "
)
if errorlevel 5 goto mtp_dflash_ngram
if errorlevel 4 goto mtp_dflash
if errorlevel 3 goto mtp_off
if errorlevel 2 goto mtp_tuned
:mtp_normal
set "MTP_MODE=normal"
:: stew fork crashes MTP on qwen35moe MoE (q36 / KAT)
if "%BACKEND_NAME%"=="stew" if "%MODEL%"=="%MODEL_IQ%" goto mtp_t_off
if "%BACKEND_NAME%"=="stew" if "%MODEL%"=="%MODEL_KATXS%" goto mtp_t_off
set "SPEC_FLAG=--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
:: Qwen3.8 models have one MTP layer, so normal MTP still requires n_max=1.
if "%MODEL%"=="%MODEL_Q38%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q38M%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q2X%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q3X%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
:: perf fork (rocm-perf): cap draft depth so it fits alongside the expert cache
if "%MOE_BACKEND%"=="1" if "%MODEL%"=="%MODEL_IQ%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 3 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MOE_BACKEND%"=="1" if "%MODEL%"=="%MODEL_KATXS%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
goto apply_config

:mtp_tuned
set "MTP_MODE=tuned"
:: branch per backend (mirrors run_model.sh)
if "%BACKEND_NAME%"=="stew" goto mtp_t_stew
if "%MOE_BACKEND%"=="1" goto mtp_t_moe
if "%BACKEND_NAME%"=="selfbuilt" goto mtp_t_rocm
if "%BACKEND_NAME%"=="rocm-72" goto mtp_t_rocm
if "%BACKEND_NAME%"=="rocm-official" goto mtp_t_rocm
if "%BACKEND_IS_VK%"=="1" goto mtp_t_vk
goto mtp_t_plain

:mtp_t_stew
:: stew rdna-boosts fork: adaptive spec is its signature; crashes on qwen35moe MoE (q36 / KAT)
if "%MODEL%"=="%MODEL_RIDGE%" goto mtp_ridge_adaptive
:: Qwen3.8-27B (1-layer MTP): n_max MUST be 1 (default 3 = CRASH; 2 = tg collapse 9-15)
if "%MODEL%"=="%MODEL_Q38%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q38M%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q3X%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q2X%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_IQ%" goto mtp_t_off
if "%MODEL%"=="%MODEL_KATXS%" goto mtp_t_off
if not defined SPEC_FLAG set "SPEC_FLAG=--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
goto apply_config

:mtp_t_off
echo   (MTP off: stew fork crashes on qwen35moe MoE)
set "SPEC_FLAG="
goto apply_config

:mtp_t_moe
:: perf fork (rocm-perf): MTP is VRAM-tight on top of the expert cache (112 slots)
if "%MODEL%"=="%MODEL_IQ%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 3 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_KATXS%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if not defined SPEC_FLAG set "SPEC_FLAG=--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
goto apply_config

:mtp_t_rocm
:: official ROCm selfbuilt (zuid patch): adaptive long drafts on the q38 family (n_min3/n_max7)
if "%MODEL%"=="%MODEL_RIDGE%" set "SPEC_FLAG=--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_KATXS%" set "SPEC_FLAG=--spec-type draft-mtp,ngram-mod --spec-draft-p-min 0.82 --spec-draft-n-max 2 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32"
if "%MODEL%"=="%MODEL_Q38%" if "%BACKEND_NAME%"=="selfbuilt" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-adaptive --spec-draft-n-min 3 --spec-draft-n-max 7 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q38%" if not "%BACKEND_NAME%"=="selfbuilt" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q38M%" if "%BACKEND_NAME%"=="selfbuilt" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-adaptive --spec-draft-n-min 3 --spec-draft-n-max 7 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q38M%" if not "%BACKEND_NAME%"=="selfbuilt" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q3X%" if "%BACKEND_NAME%"=="selfbuilt" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-adaptive --spec-draft-n-min 3 --spec-draft-n-max 7 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q3X%" if not "%BACKEND_NAME%"=="selfbuilt" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q2X%" if "%BACKEND_NAME%"=="selfbuilt" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-adaptive --spec-draft-n-min 3 --spec-draft-n-max 7 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q2X%" if not "%BACKEND_NAME%"=="selfbuilt" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if not defined SPEC_FLAG set "SPEC_FLAG=--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
goto apply_config

:mtp_t_vk
:: Vulkan backends: combined spec wins on dense Ridge/KAT
if "%MODEL%"=="%MODEL_RIDGE%" goto mtp_ridge_combined
if "%MODEL%"=="%MODEL_KATXS%" set "SPEC_FLAG=--spec-type draft-mtp,ngram-mod --spec-draft-p-min 0.82 --spec-draft-n-max 2 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32"
if "%MODEL%"=="%MODEL_Q38%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q38M%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q3X%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if "%MODEL%"=="%MODEL_Q2X%" set "SPEC_FLAG=--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
if not defined SPEC_FLAG set "SPEC_FLAG=--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
goto apply_config

:mtp_t_plain
set "SPEC_FLAG=--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
goto apply_config

:mtp_ridge_adaptive
:: adaptive MTP (stew rdna-boosts) - tg 44.4 (+55%) on ROCm. Stew-only feature.
set "SPEC_FLAG=--spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 8 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
goto apply_config

:mtp_ridge_combined
:: combined draft-mtp,ngram-mod - eaman cfg: +30% tg on dense (needs b1024/ub128 + q5_1 KV, set in config_ridge)
set "SPEC_FLAG=--spec-type draft-mtp,ngram-mod --spec-draft-p-min 0.82 --spec-draft-n-max 2 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32"
goto apply_config

:mtp_off
set "MTP_MODE=off"
set "SPEC_FLAG="
goto apply_config

:mtp_dflash
if not "%DFLASH_OK%"=="1" (
  echo   DFlash2 unavailable - needs a Qwen3.8-27B model + a b10701+ build
  timeout /t 2 /nobreak >nul
  goto menu_mtp
)
set "MTP_MODE=off"
set "DFLASH_MODE=on"
if not exist "%DFLASH_DRAFT%" (
  echo   DFlash2 draft missing: %DFLASH_DRAFT%
  set "DFLASH_MODE=off"
  timeout /t 2 /nobreak >nul
  goto menu_mtp
)
set "SPEC_FLAG=--spec-type draft-dflash -md %DFLASH_DRAFT% --spec-draft-n-max 4 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0"
goto apply_config

:mtp_dflash_ngram
if not "%DFLASH_OK%"=="1" (
  echo   DFlash2 unavailable - needs a Qwen3.8-27B model + a b10701+ build
  timeout /t 2 /nobreak >nul
  goto menu_mtp
)
set "MTP_MODE=off"
set "DFLASH_MODE=on"
if not exist "%DFLASH_DRAFT%" (
  echo   DFlash2 draft missing: %DFLASH_DRAFT%
  set "DFLASH_MODE=off"
  timeout /t 2 /nobreak >nul
  goto menu_mtp
)
set "SPEC_FLAG=--spec-type draft-dflash,ngram-mod -md %DFLASH_DRAFT% --spec-draft-n-max 4 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32"
goto apply_config

:: ── Apply per-context batch/ncmoe ──────────────────
:apply_config
:: perf fork (rocm-perf): MoE expert cache - safe slot counts tuned on this GPU.
:: 112 slots leaves room for MTP draft KV: q36 MTP n_max 3 + 112 slots = 44.7 t/s,
:: KAT MTP n_max 1 + 112 slots = 61.7 t/s. 144 slots + MTP collapses (VRAM).
set "MOE_CACHE_PROFILE="
set "MOE_CACHE_SLOTS=0"
set "MOE_CACHE_FLAG="
if "%MOE_BACKEND%"=="1" if "%MODEL%"=="%MODEL_IQ%" if exist "%~dp0traces\q36-routing.csv" set "MOE_CACHE_PROFILE=%~dp0traces\q36-routing.csv"
if "%MOE_BACKEND%"=="1" if "%MODEL%"=="%MODEL_IQ%" if exist "%~dp0traces\q36-routing.csv" set "MOE_CACHE_SLOTS=112"
if "%MOE_BACKEND%"=="1" if "%MODEL%"=="%MODEL_KATXS%" if exist "%~dp0traces\kat-routing.csv" set "MOE_CACHE_PROFILE=%~dp0traces\kat-routing.csv"
if "%MOE_BACKEND%"=="1" if "%MODEL%"=="%MODEL_KATXS%" if exist "%~dp0traces\kat-routing.csv" set "MOE_CACHE_SLOTS=112"
if defined MOE_CACHE_PROFILE set "MOE_CACHE_FLAG=--moe-cache-profile %MOE_CACHE_PROFILE% --moe-cache-slots %MOE_CACHE_SLOTS%"
if "%MODEL%"=="%MODEL_Q38%" goto config_q38
if "%MODEL%"=="%MODEL_Q38M%" goto config_q38
if "%MODEL%"=="%MODEL_Q3X%" goto config_q38
if "%MODEL%"=="%MODEL_Q2X%" goto config_q38
if "%MODEL%"=="%MODEL_Q8%" goto config_q38
if "%MODEL%"=="%MODEL_RIDGE%" goto config_ridge
if "%MODEL%"=="%MODEL_KATXS%" goto config_katxs
goto config_iq_model

:config_iq_model
if "%BACKEND_IS_VK%"=="1" goto config_vulkan_iq
goto config_rocm_iq

:config_vulkan_iq
if "%CTX_SIZE%"=="32768" (set "B=4096" & set "UB=2048" & set "NCMOE=16")
if "%CTX_SIZE%"=="65536" (set "B=4096" & set "UB=2048" & set "NCMOE=16")
if "%CTX_SIZE%"=="81920" (set "B=4096" & set "UB=2048" & set "NCMOE=16")
if "%CTX_SIZE%"=="102400" (set "B=4096" & set "UB=2048" & set "NCMOE=18")
if "%CTX_SIZE%"=="131072" (set "B=4096" & set "UB=2048" & set "NCMOE=18")
if "%CTX_SIZE%"=="153600" (set "B=4096" & set "UB=2048" & set "NCMOE=18")
if "%CTX_SIZE%"=="262144" (set "B=512" & set "UB=512" & set "NCMOE=21")
if "%CTX_SIZE%"=="184320" (set "B=512" & set "UB=512" & set "NCMOE=21")
if "%CTX_SIZE%"=="204800" (set "B=512" & set "UB=512" & set "NCMOE=21")
goto apply_dflash

:config_rocm_iq
if "%CTX_SIZE%"=="32768" (set "B=4096" & set "UB=4096" & set "NCMOE=18")
if "%CTX_SIZE%"=="65536" (set "B=4096" & set "UB=4096" & set "NCMOE=18")
if "%CTX_SIZE%"=="81920" (set "B=4096" & set "UB=4096" & set "NCMOE=18")
if "%CTX_SIZE%"=="102400" (set "B=2048" & set "UB=2048" & set "NCMOE=20")
if "%CTX_SIZE%"=="131072" (set "B=2048" & set "UB=2048" & set "NCMOE=20")
if "%CTX_SIZE%"=="153600" (set "B=2048" & set "UB=2048" & set "NCMOE=20")
if "%CTX_SIZE%"=="262144" (set "B=512" & set "UB=512" & set "NCMOE=20")
if "%CTX_SIZE%"=="184320" (set "B=512" & set "UB=512" & set "NCMOE=20")
if "%CTX_SIZE%"=="204800" (set "B=512" & set "UB=512" & set "NCMOE=20")
goto apply_dflash

:config_q38
:: dense 27B: batch>1024 collapses at 64K+ (vulkan compute buffer). 128K MUST be b512/512 (b1024/2048 = VRAM spill, tg ~11; b512 = 30-35).
if "%CTX_SIZE%"=="32768" (set "B=2048" & set "UB=2048")
if "%CTX_SIZE%"=="65536" (set "B=1024" & set "UB=2048")
if "%CTX_SIZE%"=="81920" (set "B=1024" & set "UB=2048")
if "%CTX_SIZE%"=="102400" (set "B=512" & set "UB=512")
if "%CTX_SIZE%"=="131072" (set "B=512" & set "UB=512")
if "%CTX_SIZE%"=="153600" (set "B=512" & set "UB=512")
if "%CTX_SIZE%"=="262144" (set "B=512" & set "UB=512")
if "%CTX_SIZE%"=="184320" (set "B=512" & set "UB=512")
if "%CTX_SIZE%"=="204800" (set "B=512" & set "UB=512")
set "NCMOE=0"
goto apply_dflash

:config_ridge
:: Ridge vulkan + MTP = combined spec (draft-mtp,ngram-mod) cfg from eaman's setup: b1024/ub128 + q5_1 KV + fit-target 30 + cache-ram 6000 = tg ~40 (+30% vs no-spec)
:: Ridge vulkan no-spec / stew = plain dense pattern (b2048/2048 etc)
if "%BACKEND_IS_VK%"=="1" (
  if not "%MTP_MODE%"=="tuned" goto config_ridge_base
  if "%CTX_SIZE%"=="32768" (set "B=1024" & set "UB=128")
  if "%CTX_SIZE%"=="65536" (set "B=1024" & set "UB=128")
  if "%CTX_SIZE%"=="81920" (set "B=1024" & set "UB=128")
  if "%CTX_SIZE%"=="102400" (set "B=1024" & set "UB=128")
  if "%CTX_SIZE%"=="131072" (set "B=1024" & set "UB=128")
  if "%CTX_SIZE%"=="153600" (set "B=1024" & set "UB=128")
  if "%CTX_SIZE%"=="262144" (set "B=1024" & set "UB=128")
  if "%CTX_SIZE%"=="184320" (set "B=1024" & set "UB=128")
  if "%CTX_SIZE%"=="204800" (set "B=1024" & set "UB=128")
  set "KV_K=q4_1" & set "KV_V=q4_1"
  set "CACHE_RAM=6000"
  set "NCMOE=0"
  goto apply_dflash
)
:config_ridge_base
if "%CTX_SIZE%"=="32768" (set "B=2048" & set "UB=2048")
if "%CTX_SIZE%"=="65536" (set "B=1024" & set "UB=2048")
if "%CTX_SIZE%"=="81920" (set "B=1024" & set "UB=2048")
if "%CTX_SIZE%"=="102400" (set "B=1024" & set "UB=2048")
if "%CTX_SIZE%"=="131072" (set "B=1024" & set "UB=2048")
if "%CTX_SIZE%"=="153600" (set "B=1024" & set "UB=2048")
if "%CTX_SIZE%"=="262144" (set "B=512" & set "UB=512")
if "%CTX_SIZE%"=="184320" (set "B=512" & set "UB=512")
if "%CTX_SIZE%"=="204800" (set "B=512" & set "UB=512")
set "NCMOE=0"
goto apply_dflash

:config_katxs
if "%BACKEND_IS_VK%"=="1" goto config_vulkan_katxs
goto config_rocm_katxs

:config_vulkan_katxs
:: KAT-MTP (19.7 GB, MTP head adds VRAM pressure) - ncmoe 14 best @32K, was 9 for non-MTP.
if "%CTX_SIZE%"=="32768" (set "B=4096" & set "UB=1024" & set "NCMOE=14")
if "%CTX_SIZE%"=="65536" (set "B=4096" & set "UB=1024" & set "NCMOE=14")
if "%CTX_SIZE%"=="81920" (set "B=4096" & set "UB=1024" & set "NCMOE=14")
if "%CTX_SIZE%"=="102400" (set "B=2048" & set "UB=1024" & set "NCMOE=16")
if "%CTX_SIZE%"=="131072" (set "B=2048" & set "UB=1024" & set "NCMOE=16")
if "%CTX_SIZE%"=="153600" (set "B=2048" & set "UB=1024" & set "NCMOE=16")
if "%CTX_SIZE%"=="262144" (set "B=512" & set "UB=512" & set "NCMOE=18")
if "%CTX_SIZE%"=="184320" (set "B=512" & set "UB=512" & set "NCMOE=18")
if "%CTX_SIZE%"=="204800" (set "B=512" & set "UB=512" & set "NCMOE=18")
goto apply_dflash

:config_rocm_katxs
if "%CTX_SIZE%"=="32768" (set "B=4096" & set "UB=1024" & set "NCMOE=10")
if "%CTX_SIZE%"=="65536" (set "B=4096" & set "UB=1024" & set "NCMOE=11")
if "%CTX_SIZE%"=="81920" (set "B=4096" & set "UB=1024" & set "NCMOE=11")
if "%CTX_SIZE%"=="102400" (set "B=4096" & set "UB=1024" & set "NCMOE=13")
if "%CTX_SIZE%"=="131072" (set "B=4096" & set "UB=1024" & set "NCMOE=13")
if "%CTX_SIZE%"=="153600" (set "B=4096" & set "UB=1024" & set "NCMOE=13")
if "%CTX_SIZE%"=="262144" (set "B=512" & set "UB=512" & set "NCMOE=17")
if "%CTX_SIZE%"=="184320" (set "B=512" & set "UB=512" & set "NCMOE=17")
if "%CTX_SIZE%"=="204800" (set "B=512" & set "UB=512" & set "NCMOE=17")
goto apply_dflash

:: DFlash2 draft needs ~2 GiB extra compute buffers -> force small batches
:apply_dflash
if not "%DFLASH_MODE%"=="on" goto launch
set "B=512" & set "UB=512" & set "NCMOE=0"
goto launch

:: ── Launch ─────────────────────────────────────────
:launch
cls
echo ================================================
echo  Starting llama-server...
echo ================================================
echo   Model     : %MODEL%
echo   Backend   : %SERVER%
echo   Context   : %CTX_SIZE%   Batch: %B%   UBatch: %UB%   ncmoe: %NCMOE%
echo   KV cache  : -ctk %KV_K% -ctv %KV_V%   cache-ram: %CACHE_RAM%
echo   Sampling  : %SAMPLING_FLAG%
echo   Reasoning : %Q38_FLAG%
echo   MTP/Spec  : %SPEC_FLAG%
if "%DFLASH_MODE%"=="on" echo   DFlash2   : %DFLASH_DRAFT% (draft-dflash, n-max 4, KV4)
if not defined MOE_CACHE_PROFILE echo   MoE cache : off
if defined MOE_CACHE_PROFILE echo   MoE cache : %MOE_CACHE_PROFILE% (%MOE_CACHE_SLOTS% slots)
echo   Vision    : %MMPROJ_FLAG%
echo   Fixed     : -t 8 --flash-attn on --load-mode none --no-repack --fit off -np 1 --port 8081 --jinja --log-colors on -lv 3 --log-timestamps
echo.
echo  Server: http://localhost:8081
echo  Press Ctrl+C to stop
echo ================================================
echo.

:: Kill stale process holding port 8081 (fixes "Server stopped (code 1)" on restart)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8081" ^| findstr "LISTENING"') do (
  taskkill /f /pid %%p >nul 2>&1
)

"%SERVER%" -m "%MODEL%" -c %CTX_SIZE% -t 8 -b %B% --ubatch-size %UB% ^
  -ctk %KV_K% -ctv %KV_V% --flash-attn on --load-mode none --n-cpu-moe %NCMOE% ^
  --cache-ram %CACHE_RAM% --no-repack --fit off -np 1 --host 0.0.0.0 --port 8081 --jinja --log-colors on -lv 3 --log-timestamps %SPEC_FLAG% %MMPROJ_FLAG% %SAMPLING_FLAG% %Q38_FLAG% %MOE_CACHE_FLAG%

echo.
echo Server stopped (code %errorlevel%). Returning to main menu...
echo.
timeout /t 2 /nobreak >nul
goto menu_model
