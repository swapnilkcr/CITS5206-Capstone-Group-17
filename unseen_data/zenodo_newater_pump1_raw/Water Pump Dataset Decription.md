# Sensor Data of University

## Overview
This dataset contains sensor data from a water supply system in a university from July 2023-2024. It includes Water Level Sensors, Vibration Sensors, Pressure Sensors, and Power meters. The files are structured as follows:

1. **Water Level Sensors**:
	- Water Level: Water Level in each tank
		- `Water Level Sensor - NeWater Tank.csv`
		- `Water Level Sensor - Potable Tank.csv`
  
2. **Pressure Sensors**:
	- Incoming: Incoming pressure into each pump
		- `Pressure Sensor - Potable Incoming Pump 1.csv`
		- `Pressure Sensor - NeWater Incoming Pump 1.csv`
		- `Pressure Sensor - Potable Incoming Pump 2.csv`
		- `Pressure Sensor - NeWater Incoming Pump 2.csv`
	- Outgoing: Outgoing pressure from each pump
		- `Pressure Sensor - Potable Outgoing Pump.csv`
		- `Pressure Sensor - NeWater Outgoing Pump.csv`
  
 3. **Vibration Sensors**:
	- X-Axis Speed: X-Axis Speed of each pump
		- `Vibration Sensor - Potable Pump 1 X-Axis Speed.csv`
		- `Vibration Sensor - Potable Pump 2 X-Axis Speed.csv`
		- `Vibration Sensor - NeWater Pump 1 X-Axis Speed.csv`
		- `Vibration Sensor - NeWater Pump 2 X-Axis Speed.csv`
	- X-Axis Displacement: X-Axis Displacement of each pump
		- `Vibration Sensor - Potable Pump 1 X-Axis Displacement.csv`
		- `Vibration Sensor - Potable Pump 2 X-Axis Displacement.csv`
		- `Vibration Sensor - NeWater Pump 1 X-Axis Displacement.csv`
		- `Vibration Sensor - NeWater Pump 2 X-Axis Displacement.csv`
	- Y-Axis Speed: Y-Axis Speed of each pump
		- `Vibration Sensor - Potable Pump 1 Y-Axis Speed.csv`
		- `Vibration Sensor - Potable Pump 2 Y-Axis Speed.csv`
		- `Vibration Sensor - NeWater Pump 1 Y-Axis Speed.csv`
		- `Vibration Sensor - NeWater Pump 2 Y-Axis Speed.csv`
	- Y-Axis Displacement: Y-Axis Displacement of each pump
		- `Vibration Sensor - Potable Pump 1 Y-Axis Displacement.csv`
		- `Vibration Sensor - Potable Pump 2 Y-Axis Displacement.csv`
		- `Vibration Sensor - NeWater Pump 1 Y-Axis Displacement.csv`
		- `Vibration Sensor - NeWater Pump 2 Y-Axis Displacement.csv`
	- Z-Axis Speed: Z-Axis Speed of each pump
		- `Vibration Sensor - Potable Pump 1 Z-Axis Speed.csv
		- `Vibration Sensor - Potable Pump 2 Z-Axis Speed.csv`
		- `Vibration Sensor - NeWater Pump 1 Z-Axis Speed.csv`
		- `Vibration Sensor - NeWater Pump 2 Z-Axis Speed.csv`
	- Z-Axis Displacement: Z-Axis Displacement of each pump
		- `Vibration Sensor - Potable Pump 1 Z-Axis Displacement.csv`
		- `Vibration Sensor - Potable Pump 2 Z-Axis Displacement.csv`
		- `Vibration Sensor - NeWater Pump 1 Z-Axis Displacement.csv`
		- `Vibration Sensor - NeWater Pump 2 Z-Axis Displacement.csv`
	- Temperature Sensor: Temperature of vibration sensor on each pump
		- `Vibration Sensor - NeWater Temperature.csv`
		- `Vibration Sensor - Potable Temperature.csv`

4. **Power Meters**:
	- Power: Power readings of 3 channels in each pump
		- `Power Sensor - NeWater Pump 1.csv`
		- `Power Sensor - NeWater Pump 2.csv`
		- `Power Sensor - Potable Pump 1.csv`
		- `Power Sensor - Potable Pump 2.csv`

5. **Current**:
	- Current: Current readings of 3 channels in each pump
		- `Current Sensor - NeWater Pump 1.csv`
		- `Current Sensor - NeWater Pump 2.csv`
		- `Current Sensor - Potable Pump 1.csv`
		- `Current Sensor - Potable Pump 2.csv`

6. **Energy**:
	- Energy: Energy readings of 3 channels in each pump
		- `Energy Sensor - NeWater Pump 1.csv`
		- `Energy Sensor - NeWater Pump 2.csv`
		- `Energy Sensor - Potable Pump 1.csv`
		- `Energy Sensor - Potable Pump 2.csv`


## Source of Data
The data was collected from July 2023 until July 2024.


---


## Water Level Sensors

**Files:**
- `Water Level Sensor - NeWater Tank.csv`
- `Water Level Sensor - Potable Tank.csv`

**Description:**  
These files contain the water level reading in NeWater Tank and Potable Water Tank. 
Each row represents the mean water level in millimetres for that time granularity (minute). 

| Column Name         	| Data Type           		| Description									|
|-----------------------|---------------------------|-----------------------------------------------|
| `Time` 				| Date(DD/MM/YYYY HH:mm:ss) | Time in which the water level was measured. 	|
| `Water Level (mm)`	| Integer 					| Water level measured in mm (millimetres). 	|


---


## Pressure Sensors

**Files:**
- `Pressure Sensor - Potable Incoming Pump 1.csv`
- `Pressure Sensor - NeWater Incoming Pump 1.csv`
- `Pressure Sensor - Potable Incoming Pump 2.csv`
- `Pressure Sensor - NeWater Incoming Pump 2.csv`
- `Pressure Sensor - Potable Outgoing Pump.csv`
- `Pressure Sensor - NeWater Outgoing Pump.csv`

**Description:**  
These files contain the pressure reading of the pipe going in and out of NeWater pumps and Potable Water pumps. 
Each row represents the mean pressure level in bars for that time granularity (minute). 

| Column Name         	| Data Type           		| Description									|
|-----------------------|---------------------------|-----------------------------------------------|
| `Time` 				| Date(DD/MM/YYYY HH:mm:ss) | Time in which the pressure was measured.	 	|
| `Pressure`			| Float	 					| Pressure level measured in bar. 				|


---


## Vibration Sensors

**Files:**
- `Vibration Sensor - Potable Pump 1 X-Axis Speed.csv`
- `Vibration Sensor - Potable Pump 2 X-Axis Speed.csv`
- `Vibration Sensor - NeWater Pump 1 X-Axis Speed.csv`
- `Vibration Sensor - NeWater Pump 2 X-Axis Speed.csv`
- `Vibration Sensor - Potable Pump 1 X-Axis Displacement.csv`
- `Vibration Sensor - Potable Pump 2 X-Axis Displacement.csv`
- `Vibration Sensor - NeWater Pump 1 X-Axis Displacement.csv`
- `Vibration Sensor - NeWater Pump 2 X-Axis Displacement.csv`
- `Vibration Sensor - Potable Pump 1 Y-Axis Speed.csv`
- `Vibration Sensor - Potable Pump 2 Y-Axis Speed.csv`
- `Vibration Sensor - NeWater Pump 1 Y-Axis Speed.csv`
- `Vibration Sensor - NeWater Pump 2 Y-Axis Speed.csv`
- `Vibration Sensor - Potable Pump 1 Y-Axis Displacement.csv`
- `Vibration Sensor - Potable Pump 2 Y-Axis Displacement.csv`
- `Vibration Sensor - NeWater Pump 1 Y-Axis Displacement.csv`
- `Vibration Sensor - NeWater Pump 2 Y-Axis Displacement.csv`
- `Vibration Sensor - Potable Pump 1 Z-Axis Speed.csv
- `Vibration Sensor - Potable Pump 2 Z-Axis Speed.csv`
- `Vibration Sensor - NeWater Pump 1 Z-Axis Speed.csv`
- `Vibration Sensor - NeWater Pump 2 Z-Axis Speed.csv`
- `Vibration Sensor - Potable Pump 1 Z-Axis Displacement.csv`
- `Vibration Sensor - Potable Pump 2 Z-Axis Displacement.csv`
- `Vibration Sensor - NeWater Pump 1 Z-Axis Displacement.csv`
- `Vibration Sensor - NeWater Pump 2 Z-Axis Displacement.csv`
- `Vibration Sensor - NeWater Temperature.csv`
- `Vibration Sensor - Potable Temperature.csv`

**Description:**  
These files contain the vibration speed, displacement, and temperature of each axis in NeWater pumps and Potable Water pumps. 
Each row represents the mean speed in millimetres per second (mm/s), mean displacement in micrometres (um), and mean temperature in degree celcius for that time granularity (minute). 

| Column Name         	| Data Type           		| Description													|
|-----------------------|---------------------------|---------------------------------------------------------------|
| `Time` 				| Date(DD/MM/YYYY HH:mm:ss) | Time in which the vibration was measured.	 					|
| `Speed`				| Float						| Speed of vibration in millimetres per second (mm/s).			|
| `Displacement`		| Float	 					| Displacement of vibration in micrometres (um). 				|
| `Temperature`			| Float						| Temperature of vibration sensor in degree celcius (°C).		|


---


## Power Meters

**Files:**
- `Power Sensor - NeWater Pump 1.csv`
- `Power Sensor - NeWater Pump 2.csv`
- `Power Sensor - Potable Pump 1.csv`
- `Power Sensor - Potable Pump 2.csv`

**Description:**  
These files contain the power reading in NeWater pumps and Potable Water pumps. 
Each row represents the mean power reading for each channel in Watts (W) for that time granularity (minute). 

| Column Name         	| Data Type           		| Description												|
|-----------------------|---------------------------|-----------------------------------------------------------|
| `Time` 				| Date(DD/MM/YYYY HH:mm:ss) | Time in which the power was measured.		 				|
| `P1`					| Float	 					| Power level of channel 1 measured in Watts (W). 			|
| `P2`					| Float	 					| Power level of channel 2 measured in Watts (W). 			|
| `P3`					| Float	 					| Power level of channel 3 measured in Watts (W). 			|


---


## Current

**Files:**
- `Current Sensor - NeWater Pump 1.csv`
- `Current Sensor - NeWater Pump 2.csv`
- `Current Sensor - Potable Pump 1.csv`
- `Current Sensor - Potable Pump 2.csv`

**Description:**  
These files contain the current reading in NeWater pumps and Potable Water pumps. 
Each row represents the mean current reading for each channel in Amperes (A) for that time granularity (minute). 

| Column Name         	| Data Type           		| Description												|
|-----------------------|---------------------------|-----------------------------------------------------------|
| `Time` 				| Date(DD/MM/YYYY HH:mm:ss) | Time in which the current was measured.		 			|
| `I1`					| Float	 					| Current of channel 1 measured in Amperes (A). 			|
| `I2`					| Float	 					| Current of channel 2 measured in Amperes (A). 			|
| `I3`					| Float	 					| Current of channel 3 measured in Amperes (A). 			|


---


## Energy

**Files:**
- `Energy Sensor - NeWater Pump 1.csv`
- `Energy Sensor - NeWater Pump 2.csv`
- `Energy Sensor - Potable Pump 1.csv`
- `Energy Sensor - Potable Pump 2.csv`

**Description:**  
These files contain the energy consumed in NeWater pumps and Potable Water pumps. 
Each row represents the cumulative energy consumed for each channel in Kilowatt hours (kWh) for that time granularity (minute). 

| Column Name         	| Data Type           		| Description												|
|-----------------------|---------------------------|-----------------------------------------------------------|
| `Time` 				| Date(DD/MM/YYYY HH:mm:ss) | Time in which the energy was recorded.		 			|
| `Em1`					| Float	 					| Cumulative energy of channel 1 in Kilowatt hours (kWh). 	|
| `Em2`					| Float	 					| Cumulative energy of channel 2 in Kilowatt hours (kWh). 	|
| `Em3`					| Float	 					| Cumulative energy of channel 3 in Kilowatt hours (kWh). 	|



---


## Additional Information

| **Event Type**      | **Event**  | **Description**                                                                                                                                                                                                 | **System**     | **Duration**                                                                                                                                                                             |
|---------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **System Maintenance** | Tank Washing | Drop in potable tank water level reading to abnormally low level before climbing slowly over the period of a week.                                                                                              | Potable        | 2023 Aug 31 - Sep 6                                                                                                                                                                      |
| **Fault**           | Hardware   | PLC failed to receive signal to activate pump when water hits low water level threshold. Causes pump activation to kick in much later when water level is very low.                                               | Non-potable    | 2023 Aug 6<br> 2023 Sep 10, 11, 12, 13, 14, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30<br> 2023 Oct 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 23, 24, 25, 26                |
|                     | Pump       | Failure of pump causes it to stop working during activation. So pump operates for short duration and either requires multiple activations to pump the water to high level threshold or requires backup pump to kick in. | Non-potable    | 2023 Sep 8, 11, 12, 13, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30<br>2023 Oct 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30<br>2024 Mar 13 |
|                     |            |                                                                                                                                                                                                                 | Potable        | 2023 Sep 6<br>2023 Sep 9<br>2023 Dec 27                                                                                                                                                   |



---



## Known Issues
- The dataset may contain missing values due to sensor issues.
