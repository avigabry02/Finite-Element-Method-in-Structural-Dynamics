clc
close all
clear all

%% Data of the problem % POINT 1

EA = [ 6.9076e8 , 2.1362e8 , 4.1586e8]; % Axial stiffness [N]
                                        % Red, green, blue
EJ = [ 5.7380e6 , 3.5400e5 , 1.7995e6]; % FLexural stiffness [N*m^2]
m = [26.2 , 8.1 , 15.8];                % mass per unit length [kg/m]
omega_max = 20*2*pi;
SF = 2;

Lmax = zeros(3,1);
for ii=1:3
    Lmax(ii) = sqrt(((pi^2)/(SF*omega_max))*sqrt(EJ(ii)/m(ii)));
end

% load of the structure
[file_i,xy,nnod,sizee,idb,ndof,incid,l,gamma,m,EA,EJ,posiz,nbeam,pr]=loadstructure;

% drawing of the structure UNDEFORMED
dis_stru(posiz,l,gamma,xy,pr,idb,ndof);

%% Assemble mass and stiffness matrix [M, K] % POINT 2
[M,K] = assem(incid,l,m,EA,EJ,gamma,idb);

MFF = M(1:ndof, 1:ndof); % extracts free-free mass matrix
KFF = K(1:ndof, 1:ndof); % extracts free-free stiffness matrix

% Compute eigenfrequency and mode shapes
[modes, omega2] = eig(MFF\KFF); % eigen modes & frequencies are calculated on the free-free matrices as they only contain the dof that are actually allowed to move/vibrate
omega = sqrt(diag(omega2));
% Sort frequencies in ascending order 
[omega,i_omega] = sort(omega);
freq0 = omega/2/pi;
% Sort mode shapes in ascending order 
modes = modes(:,i_omega);

N_modes_plot = 3; % number of modes to plot
scaling_factor = 2; % scaling factor for mode drawing

% plot all the requested mode shapes
for ii = 1:N_modes_plot
    figure;
    diseg2(modes(:,ii), scaling_factor, incid, l, gamma, posiz, idb, xy, true);
    title(['Mode shape n: ', num2str(ii), '   ' , num2str(round(omega(ii)/(2*pi),2)),' Hz']); 
    legend('undeformed','deformed')
end



%% compute the frequency response functions  % POINT 3

% node_number  dof_1 dof_2 dof_3  x_coordinate y_coordinate---> point name
% 12   0 0 0   38.0423 12.3607 --------> A
% 5   0 0 0   11.4127 3.7082 ----------> B

% coefficients of proportional damping (experimentally determined)
alpha = 0.1;
beta = 2e-4;
% proportional damping matrix as linear combination of M & K
C = alpha*M + beta*K; 
% extraction of free-free damping matrix
CFF = C(1:ndof, 1:ndof);

F0 = zeros(ndof,1); % initializing force vector (all zeros)
F0(idb(12,2)) = 1; % unitary force at force position (A)
om = (0:0.01:20)*2*pi; % creating angual frequency vector (rad/s)

for ii=1:length(om)
    A = -om(ii)^2*MFF + 1i*om(ii)*CFF + KFF; % dynamic matrix for each frequency
    X(:,ii) = A\F0; % invert the relation - X has dimensions (ndof, Nf)
end 

output1_dof = idb(12,2); % B
Y1 = X(output1_dof, :);   

output2_dof = idb(5,2); % A
Y2 = X(output2_dof, :); 

f = om / (2*pi);          % frequenze in Hz
standard_linewidth = 1.2;  % set a standard line width

% Plot 
figure;
subplot(2,1,1);
semilogy(f, abs(Y1), 'b' , 'LineWidth', standard_linewidth);
xlabel('Frequency [Hz]');
ylabel('|FRF| [m/N]');
title('FRF module ');
grid on;

subplot(2,1,2);
plot(f, unwrap(angle(Y1)), 'b' , 'LineWidth', standard_linewidth);
xlabel('Frequency [Hz]');
ylabel('Phase [rad]');
title('FRF Phase');
grid on;
sgtitle('FRF input (A) ouput (A) (collocated)')

figure;
subplot(2,1,1);
semilogy(f, abs(Y2), 'b', 'LineWidth', standard_linewidth);
xlabel('Frequency [Hz]');
ylabel('|FRF| [m/N]');
title('FRF module ');
grid on;

subplot(2,1,2);
plot(f, unwrap(angle(Y2)), 'b' , 'LineWidth', standard_linewidth);
xlabel('Frequency [Hz]');
ylabel('Phase [rad]');
title('FRF Phase');
grid on;
sgtitle('FRF input (A) ouput (B) ')


%% compute the frequency response functions using modal superposition approach % POINT 4


N_modes1 = 2; % numbers of modes used for modal estimation

% modal matrices
Phi1 = modes(:,1:N_modes1);
Mmod1 = Phi1'*MFF*Phi1;
Kmod1 = Phi1'*KFF*Phi1;
Cmod1 = Phi1'*CFF*Phi1;
Fmod1 = Phi1'*F0;

xx_mod1 = zeros(N_modes1, length(om));

for ii = 1:length(om)
    xx_mod1(:,ii) = (-om(ii)^2*Mmod1+1i*om(ii)*Cmod1+Kmod1)\Fmod1;
end

xx_m1 = Phi1*xx_mod1; % (ndof, Nf), frf for orginal nodes

% Output DOF (stesso di prima)
Ymod1 = xx_m1(output1_dof, :);

N_modes2 = 4;

% modal matrices
Phi2 = modes(:,1:N_modes2);
Mmod2 = Phi2'*MFF*Phi2;
Kmod2 = Phi2'*KFF*Phi2;
Cmod2 = Phi2'*CFF*Phi2;
Fmod2 = Phi2'*F0;

xx_mod2 = zeros(N_modes2, length(om));

for ii = 1:length(om)
    xx_mod2(:,ii) = (-om(ii)^2*Mmod2+1i*om(ii)*Cmod2+Kmod2)\Fmod2;
end

xx_m2 = Phi2*xx_mod2; % (ndof, Nf), frf for orginal nodes

% Output DOF (stesso di prima)
Ymod2 = xx_m2(output1_dof, :);

% Definisci colori per chiarezza
fem_color = 'b'; % Blu per FEM
modal_color_2modes = [0.8, 0.3, 0.0]; % Arancione per Modale 2 modi
modal_color_5modes = [0.0, 0.7, 0.7]; % Verde per Modale 5 modi (o un altro colore a tua scelta)

figure; % Crea una nuova figura per il confronto

% --- Modulo dell'FRF ---
subplot(2,1,1);
semilogy(f, abs(Y1), fem_color, 'LineWidth', standard_linewidth, 'DisplayName', 'FEM'); 
hold on; % Mantiene il grafico per aggiungere altre linee

semilogy(f, abs(Ymod1), 'LineWidth', standard_linewidth, 'Color', modal_color_2modes, 'DisplayName', ['Modal (', num2str(N_modes1), ' modes)']);
semilogy(f, abs(Ymod2), 'LineWidth', standard_linewidth, 'Color', modal_color_5modes, 'DisplayName', ['Modal (', num2str(N_modes2), ' modes)']);

xlabel('Frequency [Hz]');
ylabel('|FRF| [m/N]');
title('FRF Module');
legend('Location', 'best'); % Aggiunge una legenda con i nomi specificati
grid on;
hold off; % Rilascia il grafico

% --- Fase dell'FRF ---
subplot(2,1,2);
plot(f, unwrap(angle(Y1)), fem_color, 'LineWidth', standard_linewidth, 'DisplayName', 'FEM'); 
hold on; % Mantiene il grafico per aggiungere altre linee

plot(f, unwrap(angle(Ymod1)), 'LineWidth', standard_linewidth, 'Color', modal_color_2modes, 'DisplayName', ['Modal (', num2str(N_modes1), ' modes)']);
plot(f, unwrap(angle(Ymod2)), 'LineWidth', standard_linewidth, 'Color', modal_color_5modes, 'DisplayName', ['Modal (', num2str(N_modes2), ' modes)']);

xlabel('Frequency [Hz]');
ylabel('Phase [rad]');
title('FRF Phase');
legend('Location', 'best'); % Aggiunge una legenda con i nomi specificati
grid on;
hold off; % Rilascia il grafico

% --- Titolo Generale della Figura ---
sgtitle('FRF Comparison: FEM vs. Modal Approaches');

%% compute the static deformation due to gravity load  % POINT 5

g = 9.81; % accelleration of gravity

acc_vector = zeros(nnod*3, 1); % initializing acc vector 90x1
acc_vector(idb(:,2)) = -g; % imposing -g on every vertical dof

Fg = M*acc_vector; % create the force vector from mass matrix and accelleration vector
Fg_FF= Fg(1:ndof); % extract load vector associate only to free dof

Fg1 = MFF*acc_vector(1:ndof);

U1 = KFF \ Fg1; % compute deformation vector
U2 = KFF \ Fg_FF;


% Plot static deformation
figure;
static_scaling_factor = 300; % scaling factor for static deformation ploy
diseg2(U1,static_scaling_factor ,incid,l,gamma,posiz,idb,xy,true);  
title(['Static deformation due to gravity load, scaling factor: ' , num2str(static_scaling_factor )]);
legend('undeformed','deformed')
hold on;
static_scaling_factor = 300; % scaling factor for static deformation ploy
diseg2(U2,static_scaling_factor ,incid,l,gamma,posiz,idb,xy,true);  

[~, max_idx] = max(abs(U1)); % find index of largest deformation
U_max = U1(max_idx); % find largest deformation

% Compute total mass of the structure
Mass_tot = sum(l.*m);
% Compute the maximum vertical deflection
max_defl = U_max*1000; 
fprintf('maximum vertical deflection: %.5f mm\n', max_defl);
fprintf('Total mass of the structure: %.5f Kg\n', Mass_tot);

%% Moving load % POINT 6 (Optional point 1)

element_start = 5;
element_finish = 11;
Incid_sectioned = incid(element_start:element_finish,:);
l_sectioned = l(element_start:element_finish);
Nelements = length(Incid_sectioned(:,1)); % number of elements over which the load travels

angle= deg2rad(90-18); % 18 degrees inclination of the structure
Ldist =4*Nelements; %[m] length of travel distance
V_M = 2; %[m/s] payload velocity
%M = 1; % [Kg] mass of payload (unitary to get scalable FRF)
Fm = -500; %-1*M*g; %[N] vertical force due to payload

Fmy = Fm*sin(angle);
Fmx = Fm*cos(angle);

Tstruct = Ldist/V_M; %[s] total time over the structure
T = Tstruct*1.5; %[s] time also including falling off the structure
dt = 0.001; %[s] delta time
dx = V_M*dt; %[m] delta space 


Payload_line_x_coord = [xy(5,1)]; %load coordinate vector (just for plotting) inicialized with load initial x coordinate
Fn_global = [];
Fn_local_plot = [];

c = cos(deg2rad(18));
s = sin(deg2rad(18));

RotMatrix = [ c  s  0   0  0  0;
     -s  c  0   0  0  0;
      0  0  1   0  0  0;
      0  0  0   c  s  0;
      0  0  0  -s  c  0;
      0  0  0   0  0  1 ];


for ii=1:Nelements %loop on the elements over which the payload passes
    L = l_sectioned(ii);
    Fn_element = zeros(ndof,length(0:dx:l_sectioned(ii)));
    Fn_element_plot = zeros(ndof,length(0:dx:l_sectioned(ii)));
    counter = 1;
    for jj=0:dx:l_sectioned(ii) % loop over the lenght of each element
        a = jj;
        b = L-a;
        F_element = [Fmx*b/L,Fmy*((a*b*(b-a))/(L)^3 + b/L) , Fmy*a*b^2/(L)^2 , Fmx*a/L , Fmy*(a*b*(a-b)/(L)^3 + a/L) , -1*Fmy*a^2*b/(L)^2];
        F_local_to_global = zeros(ndof , 1);
        F_local_to_global(Incid_sectioned(ii,:)) = RotMatrix'*F_element'; % trasformiamo in coordinate globali
        F_local = zeros(ndof , 1);
        F_local(Incid_sectioned(ii,:)) = F_element;
        Fn_element(:,counter) = F_local_to_global;
        Fn_element_plot(:,counter) = F_local;
        counter = counter+1;
        Payload_line_x_coord = [Payload_line_x_coord, Payload_line_x_coord(end)+dx*cos(deg2rad(18))];
    end
    Fn_global = [Fn_global, Fn_element];
    Fn_local_plot = [Fn_local_plot, Fn_element_plot];
end
Fn_global = [Fn_global , zeros(ndof, length(0:dt:T))];
Payload_line_x_coord = [Payload_line_x_coord, zeros(1, length(0:dt:T)-1) ];
Qn_global = modes' * Fn_global;

%% Plot forces

% Esempio di dati
space = 0:dx:8;
y1 = Fn_local_plot(14, :);
y2 = Fn_local_plot(15, :);
y3 = Fn_local_plot(16, :);

% Crea un'unica figura con 3 subplot orizzontali
figure;

subplot(3,1,1);
plot(space, y1(1:length(space)), 'r', 'LineWidth', 1.5);
title('Axial force');
xlabel('length [m]');
ylabel('[N]');
grid on;

subplot(3,1,2);
plot(space, y2(1:length(space)), 'b', 'LineWidth', 1.5);
title('Shear force');
xlabel('length [m]');
ylabel('[N]');
grid on;

subplot(3,1,3);
plot(space, y3(1:length(space)), 'g', 'LineWidth', 1.5);
title('Moment');
xlabel('length [m]');
ylabel('[Nm]');
grid on;

sgtitle(['Load transfer of the node in the local system'])


%% POINT 6 part 2
time = (0:length(Qn_global)-1) * dt;

n_modes = 5;

tspan = [time(1) time(end)];

Phi = modes(:,1:n_modes);
MFF_modal = Phi'*MFF*Phi;
KFF_modal = Phi'*KFF*Phi;
CFF_modal = Phi'*CFF*Phi;

t_disp = time';% lo stesso per tutti i modi
t_vect = (1:size(Fn_global, 2))*dt;
q_disp = zeros(n_modes, length(t_vect));  % ogni riga: risposta di un modo

for ii = 1:n_modes
    q0 = [0; 0];  % stato iniziale: [q; q_dot]

    % equazione del moto per il modo ii
    f = @(t, q) [q(2); (1/MFF_modal(ii, ii)) * (-CFF_modal(ii, ii) * q(2) - KFF_modal(ii, ii) * q(1) + interp1(time, Qn_global(ii, :)', t, 'linear', 0))];

    % integrazione
    [t_sol, q_sol] = ode45(f, t_vect, q0);

    % interpolazione su time originale per allineare tutto
    %q_interp = interp1(t_sol, q_sol(:,1), time, 'linear', 0);

    % salva
    q_disp(ii, :) = q_sol(:, 1);
end

X_t = modes(:, 1:n_modes)*q_disp; % time history of the movements of all dofs
idx_A = idb(12,2);

x_A_vert = X_t(idx_A, :);

%%

figure(10);
plot(time, x_A_vert, 'b', 'LineWidth', 1);
grid on;
xlabel('Time [s]');
ylabel('Displacement');
title('Vibration of the structure apex due to the moving load');
legend('Vertical displacement at point A');



%% animation the dynamic deformation of a structure over time. POINT 6 part 3

% % --- Animation Setup ---
% figure_handle = figure(); % Create a new figure window for the animation
% ax = gca; % Get the handle to the current axes in the figure
% Payload_line_y_limits = [-5 15]; % Get current Y-axis limits for the line extent
% 
% 
% animation_scaling_factor = 2000;
% 
% % --- Video Recording Setup (Optional) ---
% % Uncomment the following lines if you wish to save the animation as a video file.
% % Adjust 'outputVideo.FrameRate' to control the speed of the saved video.
% % filename = 'dynamic_deformation_animation.avi'; % Name of the output video file
% % outputVideo = VideoWriter(filename); % Create a VideoWriter object
% % outputVideo.FrameRate = 15; % Set frame rate (e.g., 15 frames per second)
% % open(outputVideo); % Open the video file for writing
% 
% % --- Animation Loop ---
% % Iterate through columns of X_t
% frame_step = 100; % Show every 2nd frame (increase for faster playback)
% for ii = 1:frame_step:size(X_t, 2)
%     % Clear the current axes before drawing the new frame.
%     cla(ax);
% 
%     diseg2(X_t(:,ii), animation_scaling_factor, incid, l, gamma, posiz, idb, xy, true);
% 
%     payload_line = line(ax, [Payload_line_x_coord(ii) Payload_line_x_coord(ii)], Payload_line_y_limits, 'Color', 'r', 'LineWidth', 1, 'DisplayName', 'Payload Line');
%     % Set the title for the current frame.
%     title(ax, ['Dynamic Deformation at Time Step: ', num2str(ii), ...
%                ', Scaling Factor: ', num2str(animation_scaling_factor)]);
% 
%     h1 = plot(ax, NaN, NaN, 'r--', 'DisplayName', 'Undeformed');
%     h2 = plot(ax, NaN, NaN, 'b-', 'DisplayName', 'Deformed');
%     h3 = payload_line;
% 
%     legend(ax, [h1 h2 h3], {'Undeformed', 'Deformed', 'Payload Line'}, 'Location', 'northwest');
% 
% 
%     % Force MATLAB to update the figure display immediately.
%     drawnow;
% 
%     % --- Capture Frame for Video (Uncomment if saving video) ---
%     % If you uncommented the VideoWriter setup, uncomment these lines too.
%     % frame = getframe(figure_handle); % Capture the current figure as a frame
%     % writeVideo(outputVideo, frame); % Write the captured frame to the video file
% 
%     % Control Animation Speed ---
%     %pause(0.001); % Pause for 0.01 seconds per frame
% end
% 
% % --- Close Video File (Uncomment if saving video) ---
% % If you uncommented the VideoWriter setup, uncomment this line.
% % close(outputVideo);
% % disp(['Animation saved to: ', filename]);
% 
% hold(ax, 'off');
% 
% 


%% Stucture stiffening % POINT 7 (Optional point 2)

% on its dedicated matlab file