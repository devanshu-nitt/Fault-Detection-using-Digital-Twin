classdef TankDashboard < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                  matlab.ui.Figure
        StartBtn                  matlab.ui.control.Button
        PlantsensorreadingsLabel  matlab.ui.control.Label
        h3Label                   matlab.ui.control.Label
        h2Label                   matlab.ui.control.Label
        h1Label                   matlab.ui.control.Label
        G3                        matlab.ui.control.LinearGauge
        GaugeLabel_3              matlab.ui.control.Label
        G2                        matlab.ui.control.LinearGauge
        GaugeLabel_2              matlab.ui.control.Label
        Tank1sensorcmLabel        matlab.ui.control.Label
        G1                        matlab.ui.control.LinearGauge
        GaugeLabel                matlab.ui.control.Label
        ClassifieroutputLabel     matlab.ui.control.Label
        FaultLabel                matlab.ui.control.Label
        FaultStatusLabel          matlab.ui.control.Label
        FaultLamp                 matlab.ui.control.Lamp
        LampLabel                 matlab.ui.control.Label
        FaultDetectionPanel       matlab.ui.container.Panel
        FaultDropDown             matlab.ui.control.DropDown
        DropDownLabel             matlab.ui.control.Label
        SP3                       matlab.ui.control.NumericEditField
        EditField2Label_2         matlab.ui.control.Label
        h3SetpointcmLabel         matlab.ui.control.Label
        SP2                       matlab.ui.control.NumericEditField
        EditField2Label           matlab.ui.control.Label
        h2SetpointcmLabel         matlab.ui.control.Label
        SP1                       matlab.ui.control.NumericEditField
        EditFieldLabel            matlab.ui.control.Label
        h1SetpointcmLabel         matlab.ui.control.Label
        AxH3                      matlab.ui.control.UIAxes
        AxH2                      matlab.ui.control.UIAxes
        AxH1                      matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
    %% Loaded models
    TwinNet
    FaultNet
    Xps
    Yps
    Xfps            % ← ADD: fault feature normalizer params

    %% LQR controller & operating point
    K_lqr
    h0
    u0
    seq_len = 5

    %% Live plant state
    h

    %% Live twin state
    h_pred
    h_history

    %% Residual buffer for windowed statistics
    res_buffer      % ← ADD: 3×W rolling residual buffer
    W = 20          % ← ADD: window length (must match training)

    %% Simulation time counter
    t_sim

    %% Background timer
    SimTimer

    %% Physical parameters
    A1=615.75; A2=615.75; A3=615.75
    a12=5.0671; a23=5.0671; a3=5.0671
    beta12=0.9; beta23=0.8; beta3=0.3
    k1=75; k2=75; dt=0.5

    %% History logs for plotting
    TimeLog
    H1_model; H2_model; H3_model
    H1_plant; H2_plant; H3_plant
end
    
    methods (Access = private)
        
        function stepSimulation(app)
    t  = app.t_sim;
    dt = app.dt;
    H_MAX = 100;   % physical tank ceiling (cm)

    %% STEP 1 — Setpoint
    h_ref = [app.SP1.Value; app.SP2.Value; app.SP3.Value];
    h_ref = max(h_ref, [2; 2; 2]);

    %% STEP 2 — Equilibrium input
    F12e = app.beta12*app.a12*sqrt(max(h_ref(1)-h_ref(2), 0));
    F23e = app.beta23*app.a23*sqrt(max(h_ref(2)-h_ref(3), 0));
    F3e  = app.beta3 *app.a3 *sqrt(max(h_ref(3),          0));
    u_eq = [F12e/app.k1;  max((F3e - F23e)/app.k2, 0)];

    %% STEP 3 — Clean sensor (noise only)
    h_meas = app.h + 0.05*randn(3,1);

    %% STEP 4 — Fault injection onto separate copy
    h_faulty = h_meas;
    switch app.FaultDropDown.Value
        case 'Bias (h1+10)',   h_faulty(1) = h_faulty(1) + 10;
        case 'Drift (h2)',     h_faulty(2) = h_faulty(2) + 0.02*t;
        case 'Stuck (h3=20)',  h_faulty(3) = 20;
    end

    %% STEP 5 — Plant: LQR on faulty sensor
    u_plant = max(u_eq - app.K_lqr*(h_faulty - h_ref), 0);

    %% Physical plant dynamics + hard clamp (prevents h3 spike)
    h1=app.h(1); h2=app.h(2); h3=app.h(3);
    F12=app.beta12*app.a12*sqrt(max(h1-h2,0));
    F23=app.beta23*app.a23*sqrt(max(h2-h3,0));
    F3 =app.beta3 *app.a3 *sqrt(max(h3,  0));
    dh = [(app.k1*u_plant(1)-F12)/app.A1;
          (F12-F23)/app.A2;
          (F23-F3+app.k2*u_plant(2))/app.A3];
    app.h = min(max(app.h + dt*dh, 0), H_MAX);   % ← clamp 0–100 cm

    %% =====================================================
    %  STEP 6 — Digital twin: use h_meas (CLEAN sensor) as history
    %  Matches Script 1A training exactly → no steady-state offset
    %% =====================================================
    app.h_history = [h_meas, app.h_history(:,1:end-1)];  % ← h_meas, NOT h_pred

    x_in = [u_plant(1); u_plant(2);
            app.h_history(1,:)';
            app.h_history(2,:)';
            app.h_history(3,:)'];

    x_norm     = mapminmax('apply', x_in, app.Xps);
    y_norm     = app.TwinNet(x_norm);
    app.h_pred = mapminmax('reverse', y_norm, app.Yps);

    %% STEP 7 — Residual + windowed statistics
    residual = app.h_pred - h_faulty;   % ≈ 0 Normal, ≠ 0 Fault

    app.res_buffer = [residual, app.res_buffer(:, 1:end-1)];
    res_mean  = mean(app.res_buffer, 2);
    res_std   = std(app.res_buffer,  0, 2);

    x_idx   = (1:app.W)';
    x_mean  = mean(x_idx);
    denom   = sum((x_idx - x_mean).^2);
    res_slope = (app.res_buffer * (x_idx - x_mean)) / denom;

    feat_raw  = [app.h_pred; h_faulty; residual;
                 res_mean;   res_std;  res_slope];   % 18×1
    feat_norm = mapminmax('apply', feat_raw, app.Xfps);

    %% STEP 8 — Classify
    y_fault     = app.FaultNet(feat_norm);
    fault_class = vec2ind(y_fault) - 1;
    fault_names = {'Normal','Bias','Drift','Stuck'};

    %% STEP 9 — Log
    app.t_sim = t + dt;
    app.TimeLog(end+1)  = t;
    app.H1_model(end+1) = app.h_pred(1);
    app.H2_model(end+1) = app.h_pred(2);
    app.H3_model(end+1) = app.h_pred(3);
    app.H1_plant(end+1) = h_faulty(1);   % show faulty reading on plot
    app.H2_plant(end+1) = h_faulty(2);
    app.H3_plant(end+1) = h_faulty(3);

    %% STEP 10 — Plots
    N   = numel(app.TimeLog);
    win = max(1, N-200);
    tW  = app.TimeLog(win:N);

    cla(app.AxH1); hold(app.AxH1,'on');
    plot(app.AxH1, tW, app.H1_model(win:N), 'b-',  'LineWidth',1.5);
    plot(app.AxH1, tW, app.H1_plant(win:N), 'r--', 'LineWidth',1.2);
    yline(app.AxH1, h_ref(1), 'g:', 'LineWidth',1.5);
    legend(app.AxH1,'Digital Twin','Plant Sensor','Setpoint','Location','best');
    title(app.AxH1,'Tank 1 — h1'); xlabel(app.AxH1,'Time (s)'); ylabel(app.AxH1,'Level (cm)');

    cla(app.AxH2); hold(app.AxH2,'on');
    plot(app.AxH2, tW, app.H2_model(win:N), 'b-',  'LineWidth',1.5);
    plot(app.AxH2, tW, app.H2_plant(win:N), 'r--', 'LineWidth',1.2);
    yline(app.AxH2, h_ref(2), 'g:', 'LineWidth',1.5);
    legend(app.AxH2,'Digital Twin','Plant Sensor','Setpoint','Location','best');
    title(app.AxH2,'Tank 2 — h2'); xlabel(app.AxH2,'Time (s)'); ylabel(app.AxH2,'Level (cm)');

    cla(app.AxH3); hold(app.AxH3,'on');
    plot(app.AxH3, tW, app.H3_model(win:N), 'b-',  'LineWidth',1.5);
    plot(app.AxH3, tW, app.H3_plant(win:N), 'r--', 'LineWidth',1.2);
    yline(app.AxH3, h_ref(3), 'g:', 'LineWidth',1.5);
    legend(app.AxH3,'Digital Twin','Plant Sensor','Setpoint','Location','best');
    title(app.AxH3,'Tank 3 — h3'); xlabel(app.AxH3,'Time (s)'); ylabel(app.AxH3,'Level (cm)');

    %% STEP 11 — Gauges (show clean h_meas so scale is meaningful)
    app.G1.Value = min(max(h_meas(1),0),80);
    app.G2.Value = min(max(h_meas(2),0),80);
    app.G3.Value = min(max(h_meas(3),0),80);

    %% STEP 12 — Fault indicator
    app.FaultLabel.Text = ['Predicted: ', fault_names{fault_class+1}];
    if fault_class == 0
        app.FaultLamp.Color = [0.0, 0.8, 0.0];
    else
        app.FaultLamp.Color = [0.9, 0.1, 0.1];
    end

    drawnow limitrate;
end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
           %% Load digital twin artefacts
    ld = load('E:\MATLAB File\Closed Loop system\tank_models.mat');
    app.TwinNet  = ld.net_twin;
    app.Xps      = ld.Xps;
    app.Yps      = ld.Yps;
    app.K_lqr    = ld.K_lqr;
    app.h0       = ld.h0;
    app.u0       = ld.u0;
    app.seq_len  = ld.seq_len;

    %% Load fault classifier — now also loads Xfps and W
    fl = load('E:\MATLAB File\Closed Loop system\fault_models.mat');
    app.FaultNet = fl.net_fault;
    app.Xfps     = fl.Xfps;     % ← ADD
    app.W        = fl.W;        % ← ADD

    %% Set initial plant and twin states to operating point
    app.h          = app.h0;
    app.h_pred     = app.h0;
    app.h_history  = repmat(app.h0, 1, app.seq_len);
    app.res_buffer = zeros(3, app.W);   % ← ADD: initialise empty buffer
    app.t_sim      = 0;

    %% Pre-fill setpoint fields
    app.SP1.Value = app.h0(1);
    app.SP2.Value = app.h0(2);
    app.SP3.Value = app.h0(3);

    %% Clear data logs
    app.TimeLog  = [];
    app.H1_model = []; app.H2_model = []; app.H3_model = [];
    app.H1_plant = []; app.H2_plant = []; app.H3_plant = [];

    %% Initialise timer
    app.SimTimer = timer.empty;

    %% Configure axes
    axes_list = [app.AxH1, app.AxH2, app.AxH3];
    for i = 1:3
        ax = axes_list(i);
        hold(ax, 'on');
        grid(ax, 'on');
        xlabel(ax, 'Time (s)');
        ylabel(ax, 'Level (cm)');
        title(ax, ['Tank ', num2str(i), ' — h', num2str(i)]);
    end

    %% Reset status displays
    app.FaultLabel.Text = 'Predicted: —';
    app.FaultLamp.Color = [0.0, 0.8, 0.0];
    app.StartBtn.Text   = 'Start';
        end

        % Button pushed function: StartBtn
        function StartBtnButtonPushed(app, event)
            if strcmp(app.StartBtn.Text, 'Start')
                %% --- Launch simulation ---
                app.StartBtn.Text = 'Stop';
                app.SimTimer = timer( ...
                    'ExecutionMode', 'fixedRate', ...
                    'Period',        0.1, ...
                    'TimerFcn',      @(~,~) stepSimulation(app));
                start(app.SimTimer);
            else
                %% --- Halt simulation ---
                app.StartBtn.Text = 'Start';
                if ~isempty(app.SimTimer) && isvalid(app.SimTimer)
                    stop(app.SimTimer);
                    delete(app.SimTimer);
                end
                app.SimTimer = timer.empty;
            end
        end

        % Close request function: UIFigure
        function UIFigureCloseRequest(app, event)
            if ~isempty(app.SimTimer) && isvalid(app.SimTimer)
                stop(app.SimTimer);
                delete(app.SimTimer);
            end
            delete(app);
            
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1100 650];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);

            % Create AxH1
            app.AxH1 = uiaxes(app.UIFigure);
            title(app.AxH1, 'Title')
            xlabel(app.AxH1, 'X')
            ylabel(app.AxH1, 'Y')
            zlabel(app.AxH1, 'Z')
            app.AxH1.Position = [30 430 680 180];

            % Create AxH2
            app.AxH2 = uiaxes(app.UIFigure);
            title(app.AxH2, 'Title')
            xlabel(app.AxH2, 'X')
            ylabel(app.AxH2, 'Y')
            zlabel(app.AxH2, 'Z')
            app.AxH2.Position = [30 230 680 180];

            % Create AxH3
            app.AxH3 = uiaxes(app.UIFigure);
            title(app.AxH3, 'Title')
            xlabel(app.AxH3, 'X')
            ylabel(app.AxH3, 'Y')
            zlabel(app.AxH3, 'Z')
            app.AxH3.Position = [30 30 680 180];

            % Create h1SetpointcmLabel
            app.h1SetpointcmLabel = uilabel(app.UIFigure);
            app.h1SetpointcmLabel.Position = [750 580 100 22];
            app.h1SetpointcmLabel.Text = 'h1 Setpoint (cm)';

            % Create EditFieldLabel
            app.EditFieldLabel = uilabel(app.UIFigure);
            app.EditFieldLabel.HorizontalAlignment = 'right';
            app.EditFieldLabel.Position = [800 486 55 22];
            app.EditFieldLabel.Text = 'Edit Field';

            % Create SP1
            app.SP1 = uieditfield(app.UIFigure, 'numeric');
            app.SP1.Position = [860 580 80 22];
            app.SP1.Value = 48.9;

            % Create h2SetpointcmLabel
            app.h2SetpointcmLabel = uilabel(app.UIFigure);
            app.h2SetpointcmLabel.Position = [750 548 100 22];
            app.h2SetpointcmLabel.Text = 'h2 Setpoint (cm)';

            % Create EditField2Label
            app.EditField2Label = uilabel(app.UIFigure);
            app.EditField2Label.HorizontalAlignment = 'right';
            app.EditField2Label.Position = [849 387 62 22];
            app.EditField2Label.Text = 'Edit Field2';

            % Create SP2
            app.SP2 = uieditfield(app.UIFigure, 'numeric');
            app.SP2.Position = [860 548 80 22];
            app.SP2.Value = 44.4;

            % Create h3SetpointcmLabel
            app.h3SetpointcmLabel = uilabel(app.UIFigure);
            app.h3SetpointcmLabel.Position = [750 516 100 22];
            app.h3SetpointcmLabel.Text = 'h3 Setpoint (cm)';

            % Create EditField2Label_2
            app.EditField2Label_2 = uilabel(app.UIFigure);
            app.EditField2Label_2.HorizontalAlignment = 'right';
            app.EditField2Label_2.Position = [849 129 62 22];
            app.EditField2Label_2.Text = 'Edit Field2';

            % Create SP3
            app.SP3 = uieditfield(app.UIFigure, 'numeric');
            app.SP3.Position = [860 516 80 22];
            app.SP3.Value = 39.4;

            % Create DropDownLabel
            app.DropDownLabel = uilabel(app.UIFigure);
            app.DropDownLabel.HorizontalAlignment = 'right';
            app.DropDownLabel.Position = [768 366 65 22];
            app.DropDownLabel.Text = 'Drop Down';

            % Create FaultDropDown
            app.FaultDropDown = uidropdown(app.UIFigure);
            app.FaultDropDown.Items = {'None', 'Bias (h1+10)', 'Drift (h2)', 'Stuck (h3=20)'};
            app.FaultDropDown.Position = [750 442 190 22];
            app.FaultDropDown.Value = 'None';

            % Create FaultDetectionPanel
            app.FaultDetectionPanel = uipanel(app.UIFigure);
            app.FaultDetectionPanel.Title = 'Fault Detection';
            app.FaultDetectionPanel.Position = [740 200 220 170];

            % Create LampLabel
            app.LampLabel = uilabel(app.UIFigure);
            app.LampLabel.HorizontalAlignment = 'right';
            app.LampLabel.Position = [785 82 35 22];
            app.LampLabel.Text = 'Lamp';

            % Create FaultLamp
            app.FaultLamp = uilamp(app.UIFigure);
            app.FaultLamp.Position = [790 310 30 45];
            app.FaultLamp.Color = [0 0.8 0];

            % Create FaultStatusLabel
            app.FaultStatusLabel = uilabel(app.UIFigure);
            app.FaultStatusLabel.Position = [830 315 100 22];
            app.FaultStatusLabel.Text = 'Fault Status';

            % Create FaultLabel
            app.FaultLabel = uilabel(app.UIFigure);
            app.FaultLabel.FontSize = 14;
            app.FaultLabel.FontWeight = 'bold';
            app.FaultLabel.Position = [760 265 200 30];
            app.FaultLabel.Text = 'Predicted: —';

            % Create ClassifieroutputLabel
            app.ClassifieroutputLabel = uilabel(app.UIFigure);
            app.ClassifieroutputLabel.FontSize = 11;
            app.ClassifieroutputLabel.Position = [760 245 200 22];
            app.ClassifieroutputLabel.Text = 'Classifier output:';

            % Create GaugeLabel
            app.GaugeLabel = uilabel(app.UIFigure);
            app.GaugeLabel.HorizontalAlignment = 'center';
            app.GaugeLabel.Position = [808 74 41 22];
            app.GaugeLabel.Text = 'Gauge';

            % Create G1
            app.G1 = uigauge(app.UIFigure, 'linear');
            app.G1.Limits = [0 80];
            app.G1.MajorTicks = [0 20 40 60 80];
            app.G1.Position = [740 95 60 100];

            % Create Tank1sensorcmLabel
            app.Tank1sensorcmLabel = uilabel(app.UIFigure);
            app.Tank1sensorcmLabel.HorizontalAlignment = 'center';
            app.Tank1sensorcmLabel.Position = [750 155 200 22];
            app.Tank1sensorcmLabel.Text = 'Tank 1 sensor (cm)';

            % Create GaugeLabel_2
            app.GaugeLabel_2 = uilabel(app.UIFigure);
            app.GaugeLabel_2.HorizontalAlignment = 'center';
            app.GaugeLabel_2.Position = [818 30 41 22];
            app.GaugeLabel_2.Text = 'Gauge';

            % Create G2
            app.G2 = uigauge(app.UIFigure, 'linear');
            app.G2.Limits = [0 80];
            app.G2.MajorTicks = [0 20 40 60 80];
            app.G2.Position = [820 95 60 100];

            % Create GaugeLabel_3
            app.GaugeLabel_3 = uilabel(app.UIFigure);
            app.GaugeLabel_3.HorizontalAlignment = 'center';
            app.GaugeLabel_3.Position = [838 10 41 22];
            app.GaugeLabel_3.Text = 'Gauge';

            % Create G3
            app.G3 = uigauge(app.UIFigure, 'linear');
            app.G3.Limits = [0 80];
            app.G3.MajorTicks = [0 20 40 60 80];
            app.G3.Position = [900 95 60 100];

            % Create h1Label
            app.h1Label = uilabel(app.UIFigure);
            app.h1Label.Position = [750 80 40 18];
            app.h1Label.Text = 'h1';

            % Create h2Label
            app.h2Label = uilabel(app.UIFigure);
            app.h2Label.Position = [830 80 40 18];
            app.h2Label.Text = 'h2';

            % Create h3Label
            app.h3Label = uilabel(app.UIFigure);
            app.h3Label.Position = [910 80 40 18];
            app.h3Label.Text = 'h3';

            % Create PlantsensorreadingsLabel
            app.PlantsensorreadingsLabel = uilabel(app.UIFigure);
            app.PlantsensorreadingsLabel.FontWeight = 'bold';
            app.PlantsensorreadingsLabel.Position = [740 200 220 22];
            app.PlantsensorreadingsLabel.Text = 'Plant sensor readings';

            % Create StartBtn
            app.StartBtn = uibutton(app.UIFigure, 'push');
            app.StartBtn.ButtonPushedFcn = createCallbackFcn(app, @StartBtnButtonPushed, true);
            app.StartBtn.Position = [750 395 190 35];
            app.StartBtn.Text = 'Start';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = TankDashboard

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end
