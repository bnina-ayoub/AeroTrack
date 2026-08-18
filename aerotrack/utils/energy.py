#from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class EnergySummary:
    backend: str
    duration_s: float
    energy_j: float
    average_power_w: float
    peak_power_w: float
    sample_count: int


class EnergyMonitor:
    def __init__(self, sample_interval_s: float = 1.0, preferred_backend: str = "auto"):
        self.sample_interval_s = max(sample_interval_s, 0.1)
        self.preferred_backend = preferred_backend
        self.backend = self._detect_backend()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._process: Optional[subprocess.Popen] = None

    def _detect_backend(self) -> Optional[str]:
        if self.preferred_backend in {"nvidia-smi", "tegrastats"}:
            return self.preferred_backend if shutil.which(self.preferred_backend) else None

        if shutil.which("nvidia-smi"):
            return "nvidia-smi"
        if shutil.which("tegrastats"):
            return "tegrastats"
        return None

    def start(self) -> bool:
        if self.backend is None:
            logger.warning("Energy monitoring unavailable: neither nvidia-smi nor tegrastats was found.")
            return False

        self._stop_event.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, name="energy-monitor", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> Optional[EnergySummary]:
        self._stop_event.set()

        if self._process is not None:
            try:
                self._process.terminate()
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        return self.summary()

    def summary(self) -> Optional[EnergySummary]:
        if len(self._samples) < 2:
            return None

        timestamps = [sample[0] for sample in self._samples]
        powers = [sample[1] for sample in self._samples]
        duration_s = max(timestamps[-1] - timestamps[0], 0.0)
        if duration_s <= 0.0:
            return None

        energy_j = 0.0
        for index in range(1, len(self._samples)):
            t_prev, p_prev = self._samples[index - 1]
            t_curr, p_curr = self._samples[index]
            energy_j += 0.5 * (p_prev + p_curr) * max(t_curr - t_prev, 0.0)

        average_power_w = energy_j / duration_s if duration_s > 0 else 0.0
        peak_power_w = max(powers)

        return EnergySummary(
            backend=self.backend or "unknown",
            duration_s=duration_s,
            energy_j=energy_j,
            average_power_w=average_power_w,
            peak_power_w=peak_power_w,
            sample_count=len(self._samples),
        )

    def _record_sample(self, power_watts: float) -> None:
        self._samples.append((time.perf_counter(), power_watts))

    def _run(self) -> None:
        if self.backend == "nvidia-smi":
            self._run_nvidia_smi()
        elif self.backend == "tegrastats":
            self._run_tegrastats()

    def _run_nvidia_smi(self) -> None:
        query = [
            "nvidia-smi",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]

        while not self._stop_event.is_set():
            power_watts = self._sample_nvidia_smi(query)
            if power_watts is not None:
                self._record_sample(power_watts)
            self._stop_event.wait(self.sample_interval_s)

    def _sample_nvidia_smi(self, query) -> Optional[float]:
        try:
            output = subprocess.check_output(query, stderr=subprocess.DEVNULL, text=True, timeout=5)
        except Exception:
            return None

        values: list[float] = []
        for line in output.splitlines():
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
            if match is not None:
                values.append(float(match.group(1)))

        if not values:
            return None
        return sum(values) / len(values)

    def _run_tegrastats(self) -> None:
        try:
            self._process = subprocess.Popen(
                ["tegrastats"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            logger.warning(f"Unable to start tegrastats: {exc}")
            return

        if self._process.stdout is None:
            return

        try:
            while not self._stop_event.is_set():
                line = self._process.stdout.readline()
                if not line:
                    if self._process.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue

                power_watts = self._parse_tegrastats_power(line)
                if power_watts is not None:
                    self._record_sample(power_watts)
        finally:
            try:
                if self._process.poll() is None:
                    self._process.terminate()
                    self._process.wait(timeout=2.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass

    def _parse_tegrastats_power(self, line: str) -> Optional[float]:
        preferred_tokens = ("VDD_IN", "POM_5V_IN", "VDD_CPU_GPU_CV", "VDD_SOC")
        for token in preferred_tokens:
            match = re.search(rf"{token}\s+([0-9]+(?:\.[0-9]+)?)", line)
            if match is not None:
                return float(match.group(1)) / 1000.0
        return None