# ADI

This project is designed to evaluate the protocol provided in "Ambit of Data Integrity- A Hierarchical Approach for the Internet of Things". It introduces 4 levels of data integirity and proposed a method based on contextual data integirty to choose between layers based on the verification requirements and contextula attrebiutes. The framework uses a combination of SDN and SDP. The protocol flow is desxribed below:

Servers and services are authenticated and registered in an SDP controller before the process begins. Each device generating data first communicates with the owner's gateway (SDN switch called OG here). Data and meta data (update frequency, lifetime, tolerable latency and ID of the destination cloud server) reach the OG. 2) OG communicates this with the SDN controller to find the route to the SDP controller (CTRL). 3) The SDN controller installs the proper rules for the route between the OG and CTRL. 4) The OG sends SPA (single packet authorization – used for identifying clients in SDP) to CTRL. 5) CTRL authenticates OG and provides information about authorized services and connection to the AH. 6) A questionnaire may be sent to the owner to gather information about owner permission (by providing the type of information and channel and server confidentiality), data merit (by asking about how much financial or health damage would happen in the case of data loss) and storage affordability (e.g., price of storage for each ADI method). 7) Owner responds. 8) The owner response along with the service and connection information is sent to the SDN controller to decide what ADI layer to use and the corresponding routing. 9) The SDN controller forwards the information to an ADI application. The ADI application uses the decision tree to choose the suitable ADI layer. 10) The application responds with the required ADI level. 11) The SDN controller installs the rules (includes the required level of ADI and route to AH) in the corresponding SDN switches (e.g., OG). 12) OG sends SPA and request connection to the cloud server(s). 13,14) AH opens the connection to the cloud server and responds to OG. 15) OG and cloud server(s) can now exchange information for the session (16). The data retrieval process is pretty much the same; however, when a third party wants to retrieve the data (user gateway - UG instead of OG) the SDP controller should communicate with OG to request permission to share the information with the UG. Once the third party retrieves the data, OG has to verify the integrity using the correct ADI.
Running a simple simulation (on Mininet 2.2.0, Ubuntu 14.04.6 operating system and Intel® Core™ i5-560M Processor 2.66 GHz ), of this protocol using POX SDN controller and developing a python ADI selection application (python 3.4.3), shows that the whole process of sending gathered information from OG to the controller and then to ADI application to select the proper ADI layer, and sending back the chosen ADI to OG, takes $37$ ms in average. Codes are available in this directory and implementation details are described in the follwoing. 

## Files and folders specifications

pox-messenger folder: This folder includes all the files in the pox/messenger directory in pox controller. It implements "ADI application". The decision tree is implemented in "MessageReceived class" in "init.py". 

startl2.py calls the required modules form pox controller and also run pox messanger. The applictaion is implemented on pox messanger and is exchanging messages with SDN swicth through pox controller. 

test_s1.py is implementing the controller agent on SDN switch. It is run by the switch to send contextual information to the ADI application through pox controller. 

ADI.ipynb is a jupyter notebook file for analysing the dataset and choose a proper verifictaion method using ADI decision tree. The data set is provided by Zhongheng Zhang et al. and is available at: https://physionet.org/content/heart-failure-zigong/1.2/

## Execution guidance

Install POX controller https://github.com/noxrepo/pox

Install mininet

Copy the file "startl2.py" in pox/ext folder

Replace the files in folder messenger in pox/pox with the files inside pox-messeenger folder.

Run pox as `./pox.py startl2`

Run mininet as `sudo mn --topo linear,4 --controller remote`

In mininet run `s1 xterm`, it will open another terminal for you. download the test_s1.py file and run it in this xterm terminal as `python3 test_s1.py`.

