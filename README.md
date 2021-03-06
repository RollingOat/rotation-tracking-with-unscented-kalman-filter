# Airborne Robot Rotation Tracking with Unscented Kalman Filter

## Estimated Quaternion and Angular Velocity
<img src="https://user-images.githubusercontent.com/97129990/167034333-05613175-b386-4ba1-a449-72a128b9f4e3.jpg" width="300" height="300"><img src="https://user-images.githubusercontent.com/97129990/167034337-1a0d38cc-de7c-4b98-8534-c7fbe1e297d9.jpg" width="300" height="300">


## True Quaternion and Angular Velocity from Vicon System
<img src="https://user-images.githubusercontent.com/97129990/167034403-394dcd77-879b-4032-a27d-0c0629d1e9dd.jpg" width="300" height="300"><img src="https://user-images.githubusercontent.com/97129990/167034408-0d9eb887-7ef9-4d9b-8a2e-0905d223da56.jpg" width="300" height="300">


## Some thought on why Kalman filter would work
1. Trackable influence: Intuitively, our measurement should reflect the change of state given system dynamics and known external influence
2. Tntrackable influence: there are some factors which aren't taken into consideration. Those factors are described as noise when incorporate observation/measurements into prediction. 
3. Will estimation accuracy be improved if we make multiple measurements?: 
