

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
 * 
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
#include <utility/imumaths.h>
#include <SD.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"

//BT
//#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
//#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
//#endif
//BluetoothSerial SerialBT;

//SD File
File myfile;

bool append_success;
int data_uart;
long Amplitude;
float A_x,A_y,A_z,G_x,G_y,G_z,E_x,E_y,E_z,Q_w,Q_x,Q_y,Q_z;
unsigned long currentMillis; 
long loopTime;
/* Set the delay between fresh samples */
#define BNO055_SAMPLERATE_DELAY_MS (100)
Adafruit_BNO055 bno = Adafruit_BNO055();

unsigned char buffer_rx[30];
static const int spiClk = 4000000; // 4 MHz

//uninitalised pointers to SPI objects
SPIClass * hspi = NULL;
//SPIClass * vspi = NULL;

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

  //SerialBT.begin("ES_AD_BN_3"); //Bluetooth device name

  Serial.begin(115200);

  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  Serial.println("BNO055 initiation done");
  bno.setExtCrystalUse(true);
  //Serial.println("Calibration status values: 0=uncalibrated, 3=fully calibrated");

  pinMode(4, OUTPUT); //LED1
  //pinMode(19, OUTPUT); //LED2
  //digitalWrite(19, HIGH);
  
  //initialise two instances of the SPIClass attached to VSPI and HSPI respectively
  //vspi = new SPIClass(VSPI);
  hspi = new SPIClass(HSPI);

  //initialise vspi with default pins
  //SCLK = 18, MISO = 19, MOSI = 23, SS = 5        ............... For Micro SD Card
  //vspi->begin();
  //alternatively route through GPIO pins of your choice
  //vspi->begin(4, 16, 17, 5); //SCLK, MISO, MOSI, SS     //26,27,28,29 on ESP Module
  //vspi->beginTransaction(SPISettings(spiClk, MSBFIRST, SPI_MODE1));

  //initialise hspi with default pins 
  //SCLK = 14, MISO = 12, MOSI = 13, SS = 15   ...............  For ADC1298
  hspi->begin(14,12,13,15); 
  hspi->beginTransaction(SPISettings(spiClk, MSBFIRST, SPI_MODE1));

  //set up slave select pins as outputs as the Arduino API
  pinMode(15, OUTPUT); //HSPI SS
  pinMode(26, OUTPUT); //ADC Start
  pinMode(27, INPUT); //HSPI DRDY   // or: pinMode(27, INPUT_PULLUP);
  Serial.println("HSPI initiate done.");
  delay(100);
  
  //init ADC
  ADC1298_init();
  Serial.println("ADS1298 initiate done.");
  delay(10);
  //hspi->end();
  
  // Create semaphore to inform us when the timer has fired
  timerSemaphore = xSemaphoreCreateBinary();
  // Use 1st timer of 4 (counted from zero).
  // Set 80 divider for prescaler (see ESP32 Technical Reference Manual for more info).
  timer = timerBegin(0, 80, true);         //timer 0, div 80
  // Attach onTimer function to our timer.
  timerAttachInterrupt(timer, &onTimer, true);
  // Set alarm to call onTimer function every second(1000000) (value in microseconds).
  // Repeat the alarm (third parameter)
  timerAlarmWrite(timer, 50000, true);  //20Hz
  //timerAlarmWrite(timer, 20000, true);  //50Hz
  // Start an alarm
  timerAlarmEnable(timer);
  Serial.println("Timer initiate done.");

  //SD Card 
  if(!SD.begin()){
      Serial.println("Card Mount Failed");
      return;
  }
  uint8_t cardType = SD.cardType();

  if(cardType == CARD_NONE){
      Serial.println("No SD card attached");
      return;
  }

  Serial.print("SD Card Type: ");
  if(cardType == CARD_MMC){
      Serial.println("MMC");
  } else if(cardType == CARD_SD){
      Serial.println("SDSC");
  } else if(cardType == CARD_SDHC){
      Serial.println("SDHC");
  } else {
      Serial.println("UNKNOWN");
  }

  //myfile = SD.open("test.txt", FILE_WRITE);
  //myfile.println("testing");
  //myfile.close();
  //writeFile(SD, "/Testing.txt", "Testing ");
  File file = SD.open("/Testing.txt", FILE_APPEND);
  appendFile(SD, "/Testing.txt", "Testing\n", file);
  file.close();
  
}

void IRAM_ATTR ADC_Data_ISR() {
  
  detachInterrupt(ADC_Data_Ready.PIN);    // stop interrupt first
  //ADC_Data_Ready.number_active += 1;
  //ADC_Data_Ready.active = true;
  //Serial.print(ADC_Data_Ready.number_active);
  //Serial.print(" ");
  Read_ADC();
  
  if (append_success == true){
    if (ADC_Data_Ready.active == true){
      digitalWrite(4, HIGH);
      ADC_Data_Ready.active = false;
    }
    else if (ADC_Data_Ready.active == false){
      digitalWrite(4, LOW);
      ADC_Data_Ready.active = true;
    }
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
    Read_IMU(&A_x, &A_y, &A_z, &G_x, &G_y, &G_z, &Q_w, &Q_x, &Q_y, &Q_z);
    
    //SD.begin();
    File file = SD.open("/Testing.txt", FILE_APPEND);
    appendFile(SD, "/Testing.txt", String(isrCount).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(isrTime).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(A_x).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(A_y).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(A_z).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(G_x).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(G_y).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(G_z).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(Q_w).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(Q_x).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(Q_y).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(Q_z).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    
    appendFile(SD, "/Testing.txt", String(buffer_rx[2]).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(buffer_rx[3]).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
    appendFile(SD, "/Testing.txt", String(buffer_rx[4]).c_str(), file);appendFile(SD, "/Testing.txt", "\n", file);
    
    file.close();
    //SD.end();
    
    /*
    myfile = SD.open("test.txt", FILE_WRITE);
    delayMicroseconds(200); myfile.print(isrCount);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(isrTime);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(A_x);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(A_y);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(A_z);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(G_x);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(G_y);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(G_z);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(Q_w);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(Q_x);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(Q_y);delayMicroseconds(200); myfile.print(" ");
    delayMicroseconds(200); myfile.print(Q_z);delayMicroseconds(200); myfile.println("");

    delayMicroseconds(200); myfile.close();
    */
    
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



void Read_ADC() {
  //long Amp;
  digitalWrite(15, LOW);
  //hspi->begin(14,12,13,15); 
  //hspi->beginTransaction(SPISettings(spiClk, MSBFIRST, SPI_MODE1));
  delayMicroseconds(100); 
  //start the continuous reading mode
  hspi->write(0x10);           
  delayMicroseconds(100); 
  
  for (int i=0; i<27; i++){
    buffer_rx[i] = hspi->transfer(0x00);
  } 
  //Amp = int(buffer_rx[2]) * pow(2,16) + int(buffer_rx[3]) * pow(2,8) + int(buffer_rx[4]);
  //Serial.print(buffer_rx[2]);
  //Serial.print(" ");
  
  //hspi->endTransaction();
  //hspi->end();

  /*
  SD.begin();
  //vspi->beginTransaction(SPISettings(spiClk, MSBFIRST, SPI_MODE1));
  File file = SD.open("/Testing.txt", FILE_APPEND);
  appendFile(SD, "/Testing.txt", String(buffer_rx[2]).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
  appendFile(SD, "/Testing.txt", String(buffer_rx[3]).c_str(), file);appendFile(SD, "/Testing.txt", " ", file);
  appendFile(SD, "/Testing.txt", String(buffer_rx[4]).c_str(), file);appendFile(SD, "/Testing.txt", "\n", file);
  file.close();
  //vspi->endTransaction();
  SD.end();
  */
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

void Read_IMU(float *Acc_x,float *Acc_y,float *Acc_z,float *Gyro_x,float *Gyro_y,float *Gyro_z, float *Quat_w,float *Quat_x, float *Quat_y, float *Quat_z)
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
  imu::Quaternion quat = bno.getQuat();
  *Quat_w = quat.w();
  *Quat_x = quat.x();
  *Quat_y = quat.y();
  *Quat_z = quat.z();

}

///////////////////////  SD Operations  ///////////////////////////

void writeFile(fs::FS &fs, const char * path, const char * message){
    Serial.printf("Writing file: %s\n", path);

    File file = fs.open(path, FILE_WRITE);
    if(!file){
        Serial.println("Failed to open file for writing");
        return;
    }
    if(file.print(message)){
        Serial.println("File written");
    } else {
        Serial.println("Write failed");
    }
    file.close();
}

void appendFile(fs::FS &fs, const char * path, const char * message, File file){
    //Serial.printf("Appending to file: %s\n", path);

    //File file = fs.open(path, FILE_APPEND);
    if(!file){
        Serial.println("Failed to open file for appending");
        append_success = false;
        return;
    }
    if(file.print(message)){
        //Serial.println("Message appended");
        append_success = true;
    } else {
        Serial.println("Append failed");
        append_success = false;
    }
    
    //file.close();
}
