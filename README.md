# TrabajoFinGrado

Este proyecto parte de

https://github.com/grvcTeam/grvc-ual

Los cambios necesarios con este proyecto son:
- Introducir en carpeta ./ual_teleop/scripts 
--
	./ROS_Files/auto_teleop.py
	./ROS_Files/auxiliar.py

- Sustituir archivo ./robots_description/urdf/component_snippets.xacro
--
	./ROS_Files/models/componect_snippets.xacro

- Sustituir archivo ./robots_description/models/mbzirc/model.xacro
--
	./ROS_Files/models/model.xacro

Para el entrenamiento de la red es necesario los archivos en ./Red y los experimentos realizados. Introducir todos los archivos en una carpeta para el entrenamiento.

Para utilizar en un entorno con gazebo
roslaunch uav_abstraction_layer test_server.launch
rosrun ual_teleop auto_teleop.py

Las Versiones utilizadas:
- ROS - Kinetic
- Python 3.5.2
- OpenCV 4.0.0
- Numpy 1.16.2
- Tensorflow 1.13.1
- Keras 2.2.4
- Pygame 1.9.4
