/*
  Transmisor optico Morse para Arduino Uno/Nano.

  Conecta cada LED a su pin (D2 a D9) mediante SU PROPIA resistencia de
  220-330 ohmios, y el catodo de cada LED a GND. Los ocho LED transmiten
  el mismo patron en sincronía.

  Abre el Monitor Serie a 9600 baudios y selecciona "Nueva linea" o
  "Ambos NL y CR". Escribe A-Z, digitos 0-9 y espacios, y envia una linea.
  Espera a que termine antes de enviar el siguiente mensaje.
*/

#include <Arduino.h>

const uint8_t LED_PINS[] = {2, 3, 4, 5, 6, 7, 8, 9};
const size_t LED_COUNT = sizeof(LED_PINS) / sizeof(LED_PINS[0]);
const unsigned int MORSE_UNIT_MS = 200;
const size_t INPUT_BUFFER_SIZE = 96;

const char *const LETTER_MORSE[] = {
    ".-",    "-...",  "-.-.",  "-..",   ".",     "..-.",  "--.",
    "....",  "..",    ".---",  "-.-",   ".-..",  "--",    "-.",
    "---",   ".--.", "--.-",  ".-.",   "...",   "-",     "..-",
    "...-",  ".--",   "-..-",  "-.--",  "--.."};

const char *const DIGIT_MORSE[] = {
    "-----", ".----", "..---", "...--", "....-",
    ".....", "-....", "--...", "---..", "----."};

char inputBuffer[INPUT_BUFFER_SIZE];
size_t inputLength = 0;
bool inputOverflow = false;

void setAllLeds(bool enabled) {
  const uint8_t state = enabled ? HIGH : LOW;
  for (size_t index = 0; index < LED_COUNT; ++index) {
    digitalWrite(LED_PINS[index], state);
  }
}

const char *morseFor(char character) {
  if (character >= 'a' && character <= 'z') {
    character = static_cast<char>(character - 'a' + 'A');
  }
  if (character >= 'A' && character <= 'Z') {
    return LETTER_MORSE[character - 'A'];
  }
  if (character >= '0' && character <= '9') {
    return DIGIT_MORSE[character - '0'];
  }
  return nullptr;
}

void sendLetter(const char *pattern) {
  for (size_t index = 0; pattern[index] != '\0'; ++index) {
    const unsigned long duration = pattern[index] == '-'
                                       ? 3UL * MORSE_UNIT_MS
                                       : MORSE_UNIT_MS;
    setAllLeds(true);
    delay(duration);
    setAllLeds(false);

    if (pattern[index + 1] != '\0') {
      delay(MORSE_UNIT_MS);
    }
  }
}

void transmitMessage(const char *message) {
  bool hasPreviousLetter = false;
  bool wordGapPending = false;

  for (size_t index = 0; message[index] != '\0'; ++index) {
    const char character = message[index];
    if (character == ' ' || character == '\t') {
      if (hasPreviousLetter) {
        wordGapPending = true;
      }
      continue;
    }

    const char *pattern = morseFor(character);
    if (pattern == nullptr) {
      Serial.print(F("Caracter no soportado; se omite: "));
      Serial.print(character);
      Serial.println();
      continue;
    }

    if (hasPreviousLetter) {
      setAllLeds(false);
      delay((wordGapPending ? 7UL : 3UL) * MORSE_UNIT_MS);
    }

    sendLetter(pattern);
    hasPreviousLetter = true;
    wordGapPending = false;
  }

  setAllLeds(false);
  if (hasPreviousLetter) {
    Serial.println(F("Transmision terminada."));
  } else {
    Serial.println(F("No hay caracteres Morse validos."));
  }
}

void handleCompletedLine() {
  if (inputOverflow) {
    Serial.println(F("Error: mensaje demasiado largo (maximo 95 caracteres)."));
  } else if (inputLength == 0) {
    Serial.println(F("Mensaje vacio; no se transmite."));
  } else {
    inputBuffer[inputLength] = '\0';
    Serial.print(F("Transmitiendo: "));
    Serial.println(inputBuffer);
    transmitMessage(inputBuffer);
  }

  inputLength = 0;
  inputOverflow = false;
}

void setup() {
  for (size_t index = 0; index < LED_COUNT; ++index) {
    pinMode(LED_PINS[index], OUTPUT);
  }
  setAllLeds(false);

  Serial.begin(9600);
  Serial.println(F("Transmisor Morse listo."));
  Serial.println(F("Envia A-Z, 0-9 y espacios en una linea."));
}

void loop() {
  while (Serial.available() > 0) {
    const char received = static_cast<char>(Serial.read());

    if (received == '\r') {
      continue;
    }
    if (received == '\n') {
      handleCompletedLine();
      continue;
    }
    if (inputOverflow) {
      continue;
    }

    if (inputLength < INPUT_BUFFER_SIZE - 1) {
      inputBuffer[inputLength++] = received;
    } else {
      inputOverflow = true;
    }
  }
}
