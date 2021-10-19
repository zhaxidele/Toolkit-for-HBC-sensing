

/* The ESP32 has four SPi buses, however as of right now only two of
 * them are available to use, HSPI and VSPI. Simply using the SPI API 
 * as illustrated in Arduino examples will use VSPI, leaving HSPI unused.
 * 
 * However if we simply intialise two instance of the SPI class for both
 * of these buses both can be used. However when just using these the Arduino
 * way only will actually be outputting at a time.
 * 
 * Logic analyser capture is in the same folder as this example as
 * "multiple_bus_output.png"
 * created 30/04/2018 by Alistair Symonds
 */

// in my case HSPI is used to communicate with MCP3561, leaving VSPI unused.

#include "esp_system.h"
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "BluetoothSerial.h"
#include <SPI.h>
#include <ADC_HW.h>

//BT
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif
BluetoothSerial SerialBT;

int data_uart;
long Amplitude;
float A_x,A_y,A_z,G_x,G_y,G_z,E_x,E_y,E_z,Q_w,Q_x,Q_y,Q_z;
unsigned long currentMillis; 
long loopTime;
/* Set the delay between fresh samples */
#define BNO055_SAMPLERATE_DELAY_MS (100)
Adafruit_BNO055 bno = Adafruit_BNO055();

unsigned char buffer_rx[30];
static const int spiClk = 2000000; // 2 MHz

//uninitalised pointers to SPI objects
//SPIClass * vspi = NULL;
SPIClass * hspi = NULL;
struct GPIO_Ready { 
    const uint8_t PIN;
    uint32_t number_active;
    bool active;
};
GPIO_Ready ADC_Data_Ready{27, 0, false};

//Timer
hw_timer_t *timer = NULL;
volatile SemaphoreHandle_t timerSemaphore;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;
volatile uint32_t isrCounter = 0;
volatile uint32_t lastIsrAt = 0;

void IRAM_ATTR onTimer(){
  // Increment the counter and set the time of ISR
  portENTER_CRITICAL_ISR(&timerMux);
  isrCounter++;
  lastIsrAt = millis();
  portEXIT_CRITICAL_ISR(&timerMux);
  // Give a semaphore that we can check in the loop
  xSemaphoreGiveFromISR(timerSemaphore, NULL);
  // It is safe to use digitalRead/Write here if you want to toggle an output
}

void setup() {

  Serial.begin(115200);
  
  

  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    //SerialBT.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  //SerialBT.println("BNO055 initiation done");
  bno.setExtCrystalUse(true);
  //Serial.println("Calibration status values: 0=uncalibrated, 3=fully calibrated");

  //pinMode(18, OUTPUT); //LED1
  //pinMode(19, OUTPUT); //LED2
  //digitalWrite(19, HIGH);
  //digitalWrite(18, HIGH);

  pinMode(4, OUTPUT); //LED
  digitalWrite(4, HIGH);
  
  //initialise two instances of the SPIClass attached to VSPI and HSPI respectively
  hspi = new SPIClass(HSPI);
  
  //initialise hspi with default pins
  //SCLK = 14, MISO = 12, MOSI = 13, SS = 15
  hspi->begin(); 
  hspi->beginTransaction(SPISettings(spiClk, MSBFIRST, SPI_MODE1));

  //set up slave select pins as outputs as the Arduino API
  pinMode(15, OUTPUT); //HSPI SS
  pinMode(26, OUTPUT); //ADC Start
  pinMode(27, INPUT); //HSPI DRDY   // or: pinMode(27, INPUT_PULLUP);
  //SerialBT.println("HSPI initiate done.");
  Serial.println("HSPI initiate done.");
  delay(100);
  
  //init ADC
  ADC1298_init();
  //SerialBT.println("ADS1298 initiate done.");
  Serial.println("ADS1298 initiate done.");
  delay(10);
  
  // Create semaphore to inform us when the timer has fired
  timerSemaphore = xSemaphoreCreateBinary();
  // Use 1st timer of 4 (counted from zero).
  // Set 80 divider for prescaler (see ESP32 Technical Reference Manual for more info).
  timer = timerBegin(0, 80, true);         //timer 0, div 80
  // Attach onTimer function to our timer.
  timerAttachInterrupt(timer, &onTimer, true);
  // Set alarm to call onTimer function every second(1000000) (value in microseconds).
  // Repeat the alarm (third parameter)
  //timerAlarmWrite(timer, 50000, true);  //20Hz
  timerAlarmWrite(timer, 20000, true);  //50Hz
  //timerAlarmWrite(timer, 100000, true);  //10Hz
  // Start an alarm
  timerAlarmEnable(timer);


  delay(2000);
  SerialBT.begin("Wrist_2"); //Bluetooth device name

  
}

void IRAM_ATTR ADC_Data_ISR() {
  
  detachInterrupt(ADC_Data_Ready.PIN);    // stop interrupt first
  //ADC_Data_Ready.number_active += 1;
  //ADC_Data_Ready.active = true;
  //Serial.print(ADC_Data_Ready.number_active);
  //Serial.print(" ");
  Read_ADC();
  if (ADC_Data_Ready.active == true){
    digitalWrite(18, HIGH);
    ADC_Data_Ready.active = false;
  }
  else if (ADC_Data_Ready.active == false){
    digitalWrite(18, LOW);
    ADC_Data_Ready.active = true;
  }
}

// the loop function runs over and over again until power down or reset
void loop() { 

  if (xSemaphoreTake(timerSemaphore, 0) == pdTRUE){
    uint32_t isrCount = 0, isrTime = 0; 
    int ADC_Data = 0;
    // Read the interrupt  
    portENTER_CRITICAL(&timerMux);
    isrCount = isrCounter;
    isrTime = lastIsrAt;
    portEXIT_CRITICAL(&timerMux);
    Read_IMU(&A_x, &A_y, &A_z, &G_x, &G_y, &G_z);

    
    SerialBT.print(isrCount);
    SerialBT.print(" ");
    //delayMicroseconds(100); 
    SerialBT.print(isrTime);
    SerialBT.print(" ");
    //delayMicroseconds(100); 
    SerialBT.print(A_x);
    SerialBT.print(" ");
    //delayMicroseconds(100); 
    SerialBT.print(A_y);
    SerialBT.print(" ");
    //delayMicroseconds(100); 
    SerialBT.print(A_z);
    SerialBT.print(" ");
    //delayMicroseconds(100); 
    SerialBT.print(G_x);
    SerialBT.print(" ");
    //delayMicroseconds(100); 
    SerialBT.print(G_y);
    SerialBT.print(" ");
    //delayMicroseconds(100); 
    SerialBT.print(G_z);
    SerialBT.print(" ");
    //delayMicroseconds(100); 
    
    

    Serial.print(isrCount);
    Serial.print(" ");
    //delayMicroseconds(100); 
    Serial.print(isrTime);
    Serial.print(" ");
    //delayMicroseconds(100); 
    Serial.print(A_x);
    Serial.print(" ");
    //delayMicroseconds(100); 
    Serial.print(A_y);
    Serial.print(" ");
    //delayMicroseconds(100); 
    Serial.print(A_z);
    Serial.print(" ");
    //delayMicroseconds(100); 
    Serial.print(G_x);
    Serial.print(" ");
    //delayMicroseconds(100); 
    Serial.print(G_y);
    Serial.print(" ");
    //delayMicroseconds(100); 
    Serial.print(G_z);
    Serial.print(" ");
    //delayMicroseconds(100); 

    
    attachInterrupt(ADC_Data_Ready.PIN, ADC_Data_ISR, FALLING);

    blinkLED();
  
  }
}



int led_State = LOW;
void blinkLED(void)
{
  if (led_State == LOW) {
    led_State = HIGH;
  } else {
    led_State = LOW;
  }
  digitalWrite(4, led_State);
}


void Read_IMU(float *Acc_x,float *Acc_y,float *Acc_z,float *Gyro_x,float *Gyro_y,float *Gyro_z)
{
  // Possible vector values can be:
  // - VECTOR_ACCELEROMETER - m/s^2
  // - VECTOR_MAGNETOMETER  - uT
  // - VECTOR_GYROSCOPE     - rad/s
  // - VECTOR_EULER         - degrees
  // - VECTOR_LINEARACCEL   - m/s^2
  // - VECTOR_GRAVITY       - m/s^2
  imu::Vector<3> Linear_Acc = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  *Acc_x = Linear_Acc.x();
  *Acc_y = Linear_Acc.y();
  *Acc_z = Linear_Acc.z();
  imu::Vector<3> Gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  *Gyro_x = Gyro.x();
  *Gyro_y = Gyro.y();
  *Gyro_z = Gyro.z();
  //imu::Vector<3> Euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  //*Euler_x = Euler.x();
  //*Euler_y = Euler.y();
  //*Euler_z = Euler.z();
  // Quaternion data
  //imu::Quaternion quat = bno.getQuat();
  //*Quat_w = quat.w();
  //*Quat_x = quat.x();
  //*Quat_y = quat.y();
  //*Quat_z = quat.z();

}
void Read_ADC() {
  //long Amp;
  digitalWrite(15, LOW);
  delayMicroseconds(100); 
  //start the continuous reading mode
  hspi->write(0x10);           
  delayMicroseconds(100); 
  
  for (int i=0; i<27; i++){
    buffer_rx[i] = hspi->transfer(0x00);
  }
  //Amp = int(buffer_rx[2]) * pow(2,16) + int(buffer_rx[3]) * pow(2,8) + int(buffer_rx[4]);

  
  SerialBT.print(buffer_rx[2]);
  SerialBT.print(" ");
  SerialBT.print(buffer_rx[3]);
  SerialBT.print(" ");
  SerialBT.print(buffer_rx[4]);
  SerialBT.println("");
  
  
  /*
  Serial.print(buffer_rx[2]);
  Serial.print(" ");
  Serial.print(buffer_rx[3]);
  Serial.print(" ");
  Serial.print(buffer_rx[4]);
  Serial.println("");
  */
  
  
  //Serial.print(Amp);
  //Serial.println("");
 
  digitalWrite(15, HIGH);
}

void wr_reg_ADC(unsigned char addr){
 
  digitalWrite(15, LOW);
  delayMicroseconds(100);
  hspi->write(addr+ADC_WREG);  // 
  hspi->write(0x00);  // 
  delayMicroseconds(100); 
  hspi->write(ADC_Def_Reg_Set[addr]);  // 
  delayMicroseconds(100); 
  digitalWrite(15, HIGH);
  delayMicroseconds(100);
}

void ADC1298_init(){
  delay(1000); 
  
  // Reset
  digitalWrite(15, LOW);
  delayMicroseconds(100); 
  hspi->write(0x06);  // Reset
  delayMicroseconds(100); 
  digitalWrite(15, HIGH);
  hspi->endTransaction();
  delayMicroseconds(100); 

  //Stop converting
  digitalWrite(26, LOW);
  delayMicroseconds(100); 
  digitalWrite(15, LOW);
  delayMicroseconds(100); 
  hspi->write(0x0A);  // Stop Conversion
  delayMicroseconds(100); 
  digitalWrite(15, HIGH);
  delayMicroseconds(100); 
  digitalWrite(26, HIGH);

  //Stop the continuous read mode
  digitalWrite(15, LOW);
  delayMicroseconds(100); 
  hspi->write(0x11);  // Stop Read Data Continuously Mode
  delayMicroseconds(100); 
  digitalWrite(15, HIGH);
  delayMicroseconds(100);

  wr_reg_ADC(ADC_CONFIG3);          //configure each register in ADC
  wr_reg_ADC(ADC_CONFIG1);
  wr_reg_ADC(ADC_CONFIG2);
  wr_reg_ADC(ADC_LOFF);
  wr_reg_ADC(ADC_CH1SET);
  wr_reg_ADC(ADC_CH2SET);
  wr_reg_ADC(ADC_CH3SET);
  wr_reg_ADC(ADC_CH4SET);
  wr_reg_ADC(ADC_CH5SET);
  wr_reg_ADC(ADC_CH6SET);
  wr_reg_ADC(ADC_CH7SET);
  wr_reg_ADC(ADC_CH8SET);
  wr_reg_ADC(ADC_RLD_SENSP);
  wr_reg_ADC(ADC_RLD_SENSN);
  wr_reg_ADC(ADC_LOFF_SENSP);
  wr_reg_ADC(ADC_LOFF_SENSN);
  wr_reg_ADC(ADC_LOFF_FLIP);
  wr_reg_ADC(ADC_LOFF_STATP);
  wr_reg_ADC(ADC_LOFF_STATN);
  wr_reg_ADC(ADC_GPIO);
  wr_reg_ADC(ADC_PACE);
  wr_reg_ADC(ADC_RESP);
  wr_reg_ADC(ADC_CONFIG4);
  wr_reg_ADC(ADC_WCT1);
  wr_reg_ADC(ADC_WCT2);

  digitalWrite(26, LOW);
  delayMicroseconds(100); 
  digitalWrite(15, LOW);
  delayMicroseconds(100); 
  hspi->write(0x08);  // 
  delayMicroseconds(100);   
  digitalWrite(15, HIGH);
  delayMicroseconds(100); 
  digitalWrite(26, HIGH);
  delayMicroseconds(50); 
  
}
