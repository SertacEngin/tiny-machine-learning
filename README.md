In recent years, the integration of robotics into various industries has revolutionized manufacturing processes, automation, and even daily tasks. Robotic arms, in particular, have
emerged as versatile tools capable of performing a wide range of manipulation tasks with precision and efficiency. As the demand for robotic systems continues to grow, there is a pressing
need to develop robust control mechanisms that enable seamless interaction between humans
and machines.
This thesis explores the development and implementation of a control system for a robotic
arm, focusing on the utilization of embedded systems and tiny machine learning techniques to
enhance its functionality and user experience. The documentation presented herein outlines
the hardware setup, data measurement capabilities, and control methodologies employed in
the construction and operation of the robotic arm.
The hardware setup comprises 6 servo motors controlled by a motor driver interfaced with
an Arduino Uno microcontroller. This configuration allows for precise control of the robotic
arm’s movements within 3d environment. Additionally, a DC-DC Buck Converter provides
the necessary power supply to both the Arduino and the motor driver, ensuring uninterrupted
operation during task execution. Importantly, the converter also facilitates data measurement,
enabling the monitoring of input and output currents, voltages, and temperature levels critical
for system performance and safety.
Data measurement plays a crucial role in understanding and optimizing the robotic arm’s
operation. By capturing real-time operational parameters, such as current consumption and
temperature, insights can be gained into the system’s behavior and potential areas for improvement.
Furthermore, the control of the robotic arm is facilitated through code, allowing for precise
manipulation of individual servo motors. A Python script communicates with the Arduino
via a serial interface, enabling users to command specific movements using instructions. This
streamlined approach to control enhances the versatility and adaptability of the robotic arm,
making it suitable for a wide range of applications.
Through the integration of embedded systems and tiny machine learning techniques, this
thesis aims to contribute to the advancement of robotic control methodologies, ultimately
enhancing the efficiency, flexibility, and user-friendliness of robotic systems. By combining
hardware and software solutions, we endeavor to pave the way for innovative applications in
automation, manufacturing, healthcare, and beyond.
