#include "Arduino.h"

#include <assert.h>
#include <array>
#include <utility>
#include <vector>

MockSerialPort Serial;
std::vector<std::pair<uint8_t, uint8_t>> pinModes;
std::vector<std::pair<uint8_t, uint8_t>> writes;
std::vector<unsigned long> delays;

void pinMode(uint8_t pin, uint8_t mode) { pinModes.push_back({pin, mode}); }
void digitalWrite(uint8_t pin, uint8_t value) { writes.push_back({pin, value}); }
void delay(unsigned long milliseconds) { delays.push_back(milliseconds); }

#include "../../arduino_morse_leds.ino"

int main() {
  setup();
  assert(Serial.baud == 9600);
  assert(pinModes.size() == 8);
  for (uint8_t index = 0; index < 8; ++index) {
    assert(pinModes[index].first == index + 2);
    assert(pinModes[index].second == OUTPUT);
  }

  writes.clear();
  delays.clear();
  Serial.input = "a t\r\n";
  Serial.nextInput = 0;
  while (Serial.available()) {
    loop();
  }

  const std::vector<unsigned long> expectedDelays = {200, 200, 600, 1400, 600};
  assert(delays == expectedDelays);
  assert(writes.size() % 8 == 0);
  for (size_t group = 0; group < writes.size(); group += 8) {
    const uint8_t state = writes[group].second;
    for (uint8_t led = 0; led < 8; ++led) {
      assert(writes[group + led].first == led + 2);
      assert(writes[group + led].second == state);
    }
  }
  assert(Serial.output.find("a t") != std::string::npos);
  return 0;
}
