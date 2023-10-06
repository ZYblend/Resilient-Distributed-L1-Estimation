%% Plot file

map = binaryOccupancyMap(simpleMap);
robotPose = out.BicyclePose;

estimatedPose = out.estimated_pose1;
estimatedPose = permute(estimatedPose,[1 3 2]);
estimatedPose = reshape(estimatedPose,[],size(estimatedPose,2),1);
estimatedPose = estimatedPose.';

estimatedPose2 = out.estimated_pose2;
estimatedPose2 = permute(estimatedPose2,[1 3 2]);
estimatedPose2 = reshape(estimatedPose2,[],size(estimatedPose2,2),1);
estimatedPose2 = estimatedPose2.';

numRobots = size(robotPose, 2) / 3;
thetaIdx = 3;

% Translation
xyz = robotPose;
xyz(:, thetaIdx) = 0;

% Rotation in XYZ euler angles
theta = robotPose(:,thetaIdx);
thetaEuler = zeros(size(robotPose, 1), 3 * size(theta, 2));
thetaEuler(:, end) = theta;

% for k = 1:size(xyz, 1)
%     show(map)
%     hold on;
%     
%     % Plot Start Location
%     plotTransforms([startLoc, 0], eul2quat([0, 0, 0]))
%     text(startLoc(1), startLoc(2), 2, 'Start');
%     
%     % Plot Goal Location
%     plotTransforms([goalLoc, 0], eul2quat([0, 0, 0]))
%     text(goalLoc(1), goalLoc(2), 2, 'Goal');
%     
%     % Plot Robot's XY locations
%     plot(robotPose(:, 1), robotPose(:, 2), '-b')
% 
% %     % Plot estimated Robot's XY locations (L1)
% %     plot(estimatedPose(:, 1), estimatedPose(:, 2), 'or')
% 
%     % Plot estimated Robot's XY locations (dynamics)
%     plot(estimatedPose2(:, 1), estimatedPose2(:, 2), '--r')
%     
%     % Plot Robot's pose as it traverses the path
% % %     quat = eul2quat(thetaEuler(k, :), 'xyz');
% % %     plotTransforms(xyz(k,:), quat, 'MeshFilePath',...
% % %         'groundvehicle.stl');
%     
%     pause(0.01)
%     hold off;
% end

figure
show(map)
hold on;
plot(robotPose(:, 1), robotPose(:, 2), '-k','LineWidth',3);
hold on, plot(estimatedPose2(:, 1), estimatedPose2(:, 2), '--r');
hold on, plot(estimatedPose2(:, 4), estimatedPose2(:, 5),'--b');
hold on, plot(estimatedPose2(:, 7), estimatedPose2(:, 8),'--g');
hold on, plot(estimatedPose2(:, 10), estimatedPose2(:, 11),'--c');
legend('robot path','camera1','camera 2', 'camera 3', 'camera 4')
title('Path Localization')

figure
subplot(3,1,1)
plot(robotPose(:,1),'-k','LineWidth',2);
hold on, plot(estimatedPose2(:,1),'-r');
hold on, plot(estimatedPose2(:,4),'-b');
hold on, plot(estimatedPose2(:,7),'-g');
hold on, plot(estimatedPose2(:,10),'-c');
legend('robot state','camera 1', 'camera 2', 'camera 3', 'camera 4')
ylabel('x')

subplot(3,1,2)
plot(robotPose(:,2),'-k','LineWidth',2);
hold on, plot(estimatedPose2(:,2),'-r');
hold on, plot(estimatedPose2(:,5),'-b');
hold on, plot(estimatedPose2(:,8),'-g');
hold on, plot(estimatedPose2(:,11),'-c');
legend('robot state','camera 1', 'camera 2', 'camera 3', 'camera 4')
ylabel('y')

subplot(3,1,3)
plot(robotPose(:,3),'-k','LineWidth',2);
hold on, plot(estimatedPose2(:,3),'-r');
hold on, plot(estimatedPose2(:,6),'-b');
hold on, plot(estimatedPose2(:,9),'-g');
hold on, plot(estimatedPose2(:,12),'-c');
legend('robot state','camera 1', 'camera 2', 'camera 3', 'camera 4')
ylabel('\theta')
xlabel('Time')

%% calculate errors
% root mean square error
E1 = sqrt((1/3)*norm((1/size(robotPose,1))*vecnorm(estimatedPose(:,1:3)-robotPose,1))^2);
E2 = sqrt((1/3)*norm((1/size(robotPose,1))*vecnorm(estimatedPose(:,4:6)-robotPose,1))^2);
E3 = sqrt((1/3)*norm((1/size(robotPose,1))*vecnorm(estimatedPose(:,7:9)-robotPose,1))^2);
E4 = sqrt((1/3)*norm((1/size(robotPose,1))*vecnorm(estimatedPose(:,10:12)-robotPose,1))^2);
RMS_error = (E1+E2+E3+E4)/4