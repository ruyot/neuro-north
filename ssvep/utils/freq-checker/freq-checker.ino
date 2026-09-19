/*
  SSVEP Frequency Meter (robust hysteresis)
  Uses threshold based on known ADC range (you observed ~200 to ~1000).
  Prints raw Hz + smoothed Hz.
*/

const uint8_t LDR_PIN = A0;

// Based on what you observed:
const int ADC_MIN = 200;
const int ADC_MAX = 1000;

const int THRESH = (ADC_MIN + ADC_MAX) / 2;         // ~600
const int HYST   = (ADC_MAX - ADC_MIN) / 12;        // ~66 (good starting point)
// If still unstable, try /10 (~80). If missing edges, try /16 (~50).

const uint16_t SAMPLE_US = 300;  // faster sampling helps edge detection

bool stateHigh = false;
unsigned long lastRiseUs = 0;

float emaHz = 0.0f;
const float ALPHA = 0.25f;

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.print("THRESH="); Serial.print(THRESH);
  Serial.print(" HYST="); Serial.println(HYST);
}

void loop() {
  int v = analogRead(LDR_PIN);

  int upper = THRESH + HYST;
  int lower = THRESH - HYST;

  // Rising edge
  if (!stateHigh && v >= upper) {
    stateHigh = true;

    unsigned long nowUs = micros();
    if (lastRiseUs != 0) {
      unsigned long dt = nowUs - lastRiseUs;
      if (dt > 0) {
        float hz = 1000000.0f / (float)dt;

        // filter to plausible range
        if (hz >= 1.0f && hz <= 60.0f) {
          emaHz = (emaHz == 0.0f) ? hz : (ALPHA * hz + (1.0f - ALPHA) * emaHz);

          Serial.print("frq=");
          Serial.println(2*hz, 2);
        }
      }
    }
    lastRiseUs = nowUs;
  }

  // Falling edge (re-arm)
  if (stateHigh && v <= lower) {
    stateHigh = false;
  }

  delayMicroseconds(SAMPLE_US);
}