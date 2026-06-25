# Shared QEMU helpers for DominionOS launchers (run / run-test / run-bench).
#
# WHY THIS EXISTS - the "QEMU has its own CPU %" question.
# ----------------------------------------------------------------------------
# QEMU is NOT being throttled by DominionOS or these scripts: there is no taskset,
# no cgroup, no -icount, no affinity mask anywhere. What *looks* like a fixed
# CPU cap has two ordinary causes:
#
#   1. No -accel was passed, so QEMU fell back to TCG - a software instruction
#      emulator. TCG runs the guest on a single host thread and is ~10-100x
#      slower than native, so it pegs roughly one host core and no more.
#   2. No -smp was passed, so the guest has a single vCPU. DominionOS is itself a
#      single-core cooperative kernel (no AP bring-up), so even with more vCPUs
#      the OS would only run on one. A single-threaded guest physically cannot
#      consume 95% of a multi-core host - one busy vCPU on a 16-thread box is ~6%.
#
# THE FIX: enable hardware virtualization (whpx on Windows). Then that one vCPU
# runs at near-native speed and will drive its host core hard - bounded by the
# host, not by an artificial cap. We auto-detect whpx and fall back to
# multi-threaded TCG. Override with $env:DOMINION_ACCEL = 'whpx' | 'tcg' | 'tcg-mt'.

function Get-DominionAccel {
    # Explicit override wins.
    switch ($env:DOMINION_ACCEL) {
        'whpx'   { return 'whpx,kernel-irqchip=off' }
        'tcg'    { return 'tcg' }
        'tcg-mt' { return 'tcg,thread=multi' }
    }

    # Is whpx even compiled into this QEMU?
    $help = ''
    try { $help = (& qemu-system-x86_64 -accel help 2>&1 | Out-String) } catch { }
    if ($help -notmatch 'whpx') {
        Write-Host "[accel] whpx not available in this QEMU build; using multi-threaded TCG."
        return 'tcg,thread=multi'
    }

    # Compiled in - but the Windows Hypervisor Platform feature may be disabled.
    # Probe it: boot a throwaway VM with whpx and no disk. If whpx initializes,
    # SeaBIOS reaches "no bootable device" and the process keeps running (success);
    # if whpx init fails, QEMU prints an error and exits within a second.
    Write-Host "[accel] probing whpx (Windows Hypervisor Platform) ..."
    $proc = $null
    try {
        $proc = Start-Process -FilePath 'qemu-system-x86_64' -ArgumentList @(
            '-accel', 'whpx,kernel-irqchip=off', '-display', 'none', '-m', '64', '-no-reboot'
        ) -PassThru -WindowStyle Hidden
    } catch {
        Write-Host "[accel] could not launch probe; using multi-threaded TCG."
        return 'tcg,thread=multi'
    }
    $exited = $proc.WaitForExit(2500)
    if ($exited -and $proc.ExitCode -ne 0) {
        Write-Host "[accel] whpx present but failed to initialize (exit $($proc.ExitCode)); using multi-threaded TCG."
        Write-Host "[accel] to enable it: Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform (admin, then reboot)."
        return 'tcg,thread=multi'
    }
    if (-not $exited) { try { Stop-Process -Id $proc.Id -Force } catch { } }
    Write-Host "[accel] whpx available - guest runs at near-native speed."
    return 'whpx,kernel-irqchip=off'
}

# Number of vCPUs for the normal (non-validate) launchers. The default boot path is
# still single-core, so this is 1. Real SMP (AP bring-up) exists behind the `smp`
# feature and is exercised by run-validate.ps1, which sets its own -smp from the host
# core count; once SMP is promoted to the default boot path, return that here too.
function Get-DominionVcpus {
    return 1
}
