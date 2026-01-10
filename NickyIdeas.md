1: Lower fps livstream for rerun
    UDP instead of ZMQ using TCP FROM THE ROBOT ITSELF, then view from client.

    For rerun maybe using udp rather then zmq->tcp to send camera images would be better, we don't really have to get all the camera images just the current camera image.

    But there are things I don't know. I dont know how the camera images published on rerun are used and if they need zmq. Adn i dont even know how much performance I would gain.yeah


    no --> dont put rerun on the robot, the problem isnt that the data isnt being sent, its that the server can't handle displaying the amount of data, so use throttleing: 
    Result:
    Frame 1: Send ✅
    Frame 2-10: Skip ❌❌❌❌❌❌❌❌❌
    Frame 11: Send ✅ (this is the CURRENT frame, not Frame 2!)
    Frame 12-20: Skip
    Etc

    like a low fps live stream. DO THIS 


general idea; 

connect to robot using tcp/zmq
set up ros2 node, camera/joints publish, gpu subscribes

set up rerun server uses a buffer from camera and joints to make the 3d voxel map

performs 360, 360 degrees a certain amount of roatations, creates point cloud (idk what this is), and to my understanding the clip/other algo ONLY looks for the two objects I specificy, but maybe im wrong and it only save the location of the two object. makes points on the voxel map with rgb and something else.

Once the objects are found the sam algo or something segments the objects so we know exactly what they are and it saves their locations

somehow semantically maps things.

After than it runs a general process with setWay, moveWay, grab, arrive, place etc which each have their own specific process too which both change locations of waypoints in the voxel map and whatveer underlying datastructure it uses. 

For grab specifically 

