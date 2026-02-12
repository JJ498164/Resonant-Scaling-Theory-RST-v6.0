# RST Hardware Implementation: The Sentinel Watch

This directory contains the conceptual and technical framework for translating Resonant Scaling Theory (RST) v5.1 into wearable haptic hardware.

## Technical Specifications
- **Primary Resonance:** 39 Hz (Spectral Density Stability Constant)
- **Safety Lockout:** 6.1s (Topological Friction / Ignition Delay Buffer)
- **Hardware Target:** ESP32-based microcontrollers (e.g., LilyGo T-Watch, PineTime)
- **Goal:** Provide tactile grounding (Haptic Hub Gain) to reduce effective resistance ($R_{eff}$) in the neural network during high-stress states.

## Operation
The hardware monitors Heart Rate Variability (HRV). Upon detecting a "Friction Spike" (BPM > Threshold), the device initiates a 6.1s haptic lockout, preventing interaction until the system resynchronizes to 39 Hz.
from machine import Pin, PWM
import time

# --- CONFIGURATION ---
# Pin 14 is common for vibration motors on ESP32 watches
vibe_motor = PWM(Pin(14))
vibe_motor.freq(39) # The RST Constant

def apply_39hz_grounding(duration=2):
    """Activates the physical resonance pulse."""
    print("Applying 39Hz Hub Gain...")
    vibe_motor.duty(512) # 50% intensity
    time.sleep(duration)
    vibe_motor.duty(0)

def trigger_redline():
    """Initiates the 6.1s Ignition Delay."""
    print("Topological Friction Detected. Locking...")
    start = time.time()
    while (time.time() - start) < 6.1:
        # Rapid staccato pulse to signal the lock
        vibe_motor.duty(256)
        time.sleep(0.1)
        vibe_motor.duty(0)
        time.sleep(0.1)
    
    # Release signal
    apply_39hz_grounding(1)
    print("Resonance Restored.")

# Example Trigger
# trigger_redline()
{
  "modes": {
    "grounding": {
      "frequency": 39,
      "pattern": "continuous",
      "intent": "Reduce effective resistance across neural bridges"
    },
    "redline": {
      "duration": 6.1,
      "pattern": "staccato",
      "intent": "Model the ignition delay following axonal friction"
    },
    "family_sync": {
      "frequency": 39,
      "pattern": "heartbeat",
      "intent": "Maintain the Daughters' Line safety harbor"
    }
  }
}
