#include <SPI.h>
#include <Servo.h>
#include <Wire.h>
#include "Adafruit_PWMServoDriver.h"
#include "WString.h"

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

float deg_to_pulse(int deg)
{
  return deg * 2.5 + 150;

}

float pulse_to_deg(int pulse)
{
  return ((pulse - 150) / (2.5));
}

int angles[6] = {90, 78, 68, 168, 70, 73};
int old_angles[6] = {90, 78, 68, 168, 70, 73};
int angle_pos[6] = {0, 0, 0, 0, 0, 0};


char delim[] = "-";
char delim_big[] = "/";
char* vals[12];
char msg[25];
int x = 0;
int init_size = strlen(msg);



void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10);

  pwm.begin();
  pwm.setPWMFreq(50);
  while (!Serial) {
  }

  for (int i = 0; i <= 5; i++)
  {
    pwm.setPWM(i, 0, deg_to_pulse(old_angles[i]));
  }

}


void loop() {


  if (Serial.available() > 0)
  {

    String xstring = Serial.readStringUntil("/");
    xstring.toCharArray(msg, sizeof(msg));

    char *ptr2 = strtok(msg, delim_big);
    char* ptr = strtok(ptr2, delim);

    while (ptr != NULL)
    {
      vals[x++] = ptr;
      ptr = strtok(NULL, delim);
    }

    for (int i = 0; i < 6; i++)
    {
      angles[i] = atoi(vals[i]); // Convert the character to integer, in this case
    }


    for (int i = 0; i < 6; i++)
    {
      int pwm_old = (int)deg_to_pulse(old_angles[i]);
      int pwm_new = (int)deg_to_pulse(angles[i]);

       pwm.setPWM(i, 0, pwm_new);
    
      old_angles[i] = angles[i];

    }

    x = 0;
  }
}