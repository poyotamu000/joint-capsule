const int analogInPinA0 = A0;
const int analogInPinA1 = A1;
const int analogInPinA2 = A2;
const int analogInPinA3 = A3;
const int analogInPinA4 = A4;
const int analogInPinA5 = A5;
const int analogInPinA6 = A6;
const int analogInPinA7 = A7;
const int analogInPinA8 = A8;
const int analogInPinA9 = A9;
const int analogInPinA10 = A10;
const int analogInPinA11 = A11;
const int analogInPinA12 = A12;
const int analogInPinA13 = A13;
const int analogInPinA14 = A14;
const int analogInPinA15 = A15;

int sensorValue00 = 0;
int sensorValue01 = 0;
int sensorValue02 = 0;
int sensorValue03 = 0;
int sensorValue04 = 0;
int sensorValue05 = 0;
int sensorValue06 = 0;
int sensorValue07 = 0;
int sensorValue08 = 0;
int sensorValue09 = 0;
int sensorValue10 = 0;
int sensorValue11 = 0;
int sensorValue12 = 0;
int sensorValue13 = 0;
int sensorValue14 = 0;
int sensorValue15 = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  sensorValue00 = analogRead(analogInPinA0);
  sensorValue01 = analogRead(analogInPinA1);
  sensorValue02 = analogRead(analogInPinA2);
  sensorValue03 = analogRead(analogInPinA3);
  sensorValue04 = analogRead(analogInPinA4);
  sensorValue05 = analogRead(analogInPinA5);
  sensorValue06 = analogRead(analogInPinA6);
  sensorValue07 = analogRead(analogInPinA7);
  sensorValue08 = analogRead(analogInPinA8);
  sensorValue09 = analogRead(analogInPinA9);
  sensorValue10 = analogRead(analogInPinA10);
  sensorValue11 = analogRead(analogInPinA11);
  sensorValue12 = analogRead(analogInPinA12);
  sensorValue13 = analogRead(analogInPinA13);
  sensorValue14 = analogRead(analogInPinA14);
  sensorValue15 = analogRead(analogInPinA15);

  Serial.print("sensor 0 to 3 = ");
  Serial.print(sensorValue00);
  Serial.print(", ");
  Serial.print(sensorValue01);
  Serial.print(", ");
  Serial.print(sensorValue02);
  Serial.print(", ");
  Serial.println(sensorValue03);

  Serial.print("sensor 4 to 7 = ");
  Serial.print(sensorValue04);
  Serial.print(", ");
  Serial.print(sensorValue05);
  Serial.print(", ");
  Serial.print(sensorValue06);
  Serial.print(", ");
  Serial.println(sensorValue07);

  Serial.print("sensor 8 to 11 = ");
  Serial.print(sensorValue08);
  Serial.print(", ");
  Serial.print(sensorValue09);
  Serial.print(", ");
  Serial.print(sensorValue10);
  Serial.print(", ");
  Serial.println(sensorValue11);

  Serial.print("sensor 12 to 15 = ");
  Serial.print(sensorValue12);
  Serial.print(", ");
  Serial.print(sensorValue13);
  Serial.print(", ");
  Serial.print(sensorValue14);
  Serial.print(", ");
  Serial.println(sensorValue15);

  // printf("%4d, %4d, ", data)

  delay(2);
}
