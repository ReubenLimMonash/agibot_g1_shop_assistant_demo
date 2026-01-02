# Scene Setup
This guide details the setup of the item shelf, counter, and SLAM mapping for the demo.

## Item Shelf
The shelf was configured to have two levels. The items on the shelf should be arranged as follows:
- Large cylinders of potato chips on the top shelf at the left side.
- Small cylinders of potato chips on the top shelf at the right side.
- Small water bottles on the bottom shelf at the left side. 
- Bottles of 100 plus on the bottom shelf at the right side.

Note: The arrangements of these items can be changed, but these changes must also be reflected in the [MCP server script](./src/retail_demo_mcp_server.py). 

## SLAM Navigation Setup
1. Position the G1 to face the shelf from a short distance away (~43cm away). 
2. Start the G1's SLAM mapping function (follow G1's manual for this).
3. Move the G1 around the area to create a map of the area (the G1 can be moved during mapping using manual navigation from the Navigation->Monitoring page).
4. Bring the G1 back to the starting position and complete the mapping process. 
5. Edit and save the map.

It is important that the G1 starts mapping from the initial position specified (in front of the shelf, facing it), so that it establishes this initial position as the origin, (0,0,0), with heading angle 0 deg. The G1 will return to this initial position whenever it needs to return to the shelf (although this can be changed).

## Counter Position
The position of the counter needs to be configured in the [MCP server script](./src/retail_demo_mcp_server.py). Edit the _robot_runner() function to control how the G1 navigates to the counter after grabbing the item.

Tip: To get the map coordinate of a position, setup a station node at that position in the G1's map editor UI. The (x, y) coordinates of the station node is the coordinate you should use.
