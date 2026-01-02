# AgiBot G1 Shopping Assistant Demo
This repo contains the code to setup the G1 shopping assistant demo. This demo showcases how the G1 can be deployed to assist shoppers to grab items from a shelf. The G1 uses ARuCo marker detection to locate items on a shelf for precise grabbing. It uses SLAM navigation to navigate between the shelf to grab items and the counter to place items. The items available on the shelf must be pre-configured, as well as the location (SLAM coordinates) of the counter. The G1 should have its starting position, (0, 0, 0), located at the front of the shelf where it is positioned to grab items. 

## Scene Setup
Refer to the [Scene Setup Guide](./SCENE_SETUP.md) to setup the item shelf, G1 map (for SLAM navigation), and the counter for placing the items.

## Running The Demo
There are three things to run for this demo:
1. MCP server. This server acts as the robot controller and exposes MCP tools that can instruct the G1 to grab certain items.
2. Flask server. This server is for streaming the video frames from the head and the left and right hands.
3. n8n server. This is to run the n8n workflow that allows the users to interact with the application via a chatbot. The chatbot will call the MCP tools based on the user's requests. 

First, source the a2d_sdk environment, following the instructions in the AgiBot G1 GDK guide. 

Then, install the following Python libraries in the environment you are using to host the MCP and Flask server.
```
pip install fast-mcp, Flask, opencv-python
```

Also, put the G1 in Co-Pilot mode:
```
cd ~/a2d_sdk
robot-service -s -c ./conf/copilot.pbtxt
```

### MCP Server
After configuring the [MCP server script](./src/retail_demo_mcp_server.py), run it in a terminal:
```
python ./src/retail_demo_mcp_server.py
```

### Flask Camera Streaming
Run in another terminal:
```
python ./src/flask_camera_display.py
```
### n8n Chatbot Server
Run n8n in another terminal. Create a new workflow and import the [G1 Demo Workflow](./src/AgiBot%20G1%20Demo.json). Configure the following nodes in the workflow:
- OpenAI Chat Model - to use the OpenAI LLM model that you want.
- MCP Client - Setup MCP configuration to connect to the MCP server running. Make sure the URL path is correct.

You can also configur the system message according to need.




