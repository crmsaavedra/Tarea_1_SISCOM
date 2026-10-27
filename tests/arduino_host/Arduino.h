#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string>

#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define F(value) value

struct MockSerialPort {
  std::string input;
  std::string output;
  size_t nextInput = 0;
  unsigned long baud = 0;

  void begin(unsigned long value) { baud = value; }
  int available() const { return nextInput < input.size(); }
  int read() { return input[nextInput++]; }
  void print(const char *value) { output += value; }
  void print(char value) { output += value; }
  void print(unsigned char value) { output += static_cast<char>(value); }
  void println(const char *value) {
    output += value;
    output += '\n';
  }
  void println() { output += '\n'; }
};

extern MockSerialPort Serial;
void pinMode(uint8_t pin, uint8_t mode);
void digitalWrite(uint8_t pin, uint8_t value);
void delay(unsigned long milliseconds);
