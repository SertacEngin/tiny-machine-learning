#include <Wire.h>
#include <HardwareSerial.h>

#define DEV_ADDR 0x08
#define I_IN_ADDR 0x1E
#define V_IN_ADDR 0x20
#define I_OUT_ADDR 0x22
#define V_OUT_ADDR 0x24
#define TEMP_ADDR 0x26

uint8_t x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;


void setup() {
  Serial.begin(230400);
  Wire.begin();
}


void loop() {
  Wire.beginTransmission(DEV_ADDR);
  Wire.write(I_IN_ADDR);
  Wire.endTransmission();
  
  Wire.requestFrom(DEV_ADDR, 10);
  if (Wire.available()) {
    x1 = Wire.read();
    x2 = Wire.read();
    x3 = Wire.read();
    x4 = Wire.read();
    x5 = Wire.read();
    x6 = Wire.read();
    x7 = Wire.read();
    x8 = Wire.read();
    x9 = Wire.read();
    x10 = Wire.read();
    
    Serial.write(x1);
    Serial.write(x2);
    Serial.write(x3);
    Serial.write(x4);
    Serial.write(x5);
    Serial.write(x6);
    Serial.write(x7);
    Serial.write(x8);
    Serial.write(x9);
    Serial.write(x10); 
    Serial.write("\n");

   
  }

}