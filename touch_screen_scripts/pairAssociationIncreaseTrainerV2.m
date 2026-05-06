function pairAssociationIncreaseTrainerV2(setupID,monkeyID)

if ~exist('setupID','var') || isempty(setupID)
    setupID = 'xxx'; 
end
if ~exist('monkeyID','var') || isempty(monkeyID)
    monkeyID = 'xxxx'; 
end

%%% SETUP %%%

dbstop if error
% Screen('Preference','SkipSyncTests', 1);

%%% PARAMS %%%
switch setupID(1)
    case {'S','s'}
        if str2double(setupID(2:3))>0
            setupType=1;
        else
            setupType=2;
        end
    case {'T','t'}
        setupType=3;
    case {'N','n'}
        switch str2double(setupID(2:3))
            case 0
                setupType=4;
%             case {1,2}
%                 setupType=5;
            otherwise
                setupType=5;
        end
    otherwise
        setupType=0;
end
CONFIG.scale_touchscreen = 1; % 0.5 is useful for programming or debugging
switch  setupType
    case 0% surface pro6
        CONFIG.main_display_area = [0 0 2736 1824] * CONFIG.scale_touchscreen;
        touchscreenID = 'ipts 045E:001F Touchscreen';     
%         CONFIG.main_display_area = [0 0 1920 1080] * CONFIG.scale_touchscreen;
%         productName = 'Touchscreen';
        pixel2mm = 170/1824;
%         data_folder_path = '/home/monkey01/results';
    case 1%  weichensi with surface pro7
        CONFIG.main_display_area = [2736 0 4656 1080];
        touchscreenID = 'WingCool Inc. TouchScreen';
        pixel2mm = 165/1080;
    case 2% sculptor with surface pro6
        CONFIG.main_display_area = [2736 0 4656 1080];
        touchscreenID = 'Silicon Works Multi-touch SW4101C Touchscreen';
        pixel2mm = 165/1080;
    case 3% teclast x4
        CONFIG.main_display_area = [0 0 1920 1080];
        touchscreenID = 'Goodix Capacitive TouchScreen';
        pixel2mm = 145/1080;
    case 4 % weichensi with NUC10
        CONFIG.main_display_area = [1920 0 3840 1080];
        touchscreenID = 'WingCool Inc. TouchScreen';%'ILITEK ILITEK-TP'; %tengyu
        pixel2mm = 166.6/1080;
%     case 5 % sculptor with NUC10
%         CONFIG.main_display_area = [1920 0 3840 1080];
%         touchscreenID = 'Silicon Works Multi-touch SW4101C';
%         pixel2mm = 165/1080;
    case 5 % weichensi with NUC10
        CONFIG.main_display_area = [1024 0 2944 1080];
        touchscreenID = 'WingCool Inc. TouchScreen';
        pixel2mm = 165/1080;
end
figfolder = '/home/xing/fruit';
water_time = 0.2;
weightON = false;
CONFIG.monkey= monkeyID;
CONFIG.main_dir = fileparts(mfilename('fullpath'));
CONFIG.results_dir = fullfile(pwd,'Results'); %'~/Programming/Matlab/Incage_training/hidden_target/Results';
CONFIG.copy_code_dir = fullfile(pwd,'Results','copy_code');
CONFIG.half_side_square = 20/pixel2mm * CONFIG.scale_touchscreen; % half the side of the target square in pixels
% CONFIG.half_side_square = 25/pixel2mm * CONFIG.scale_touchscreen; 
CONFIG.radius_onset_polygon = 1 * CONFIG.half_side_square;
CONFIG.rect_touch_half_side = 1 * CONFIG.half_side_square;
CONFIG.distance_from_center_x_y = 56/pixel2mm * CONFIG.scale_touchscreen;
CONFIG.pre_sample_period = 0.5;
CONFIG.sample_period = 1;
CONFIG.pre_test_period = 0;
CONFIG.intertrial_interval = .5; % secs
CONFIG.intertrial_time_punishment = 0; % secs
CONFIG.max_time_touch_onset = 5; %secs;
CONFIG.max_time_touch_test = 10; %secs;
CONFIG.save_results = 1;
CONFIG.n_pairs = 5;
CONFIG.n_distractors = 1; % 0 to 5
CONFIG.list_objs = {'square', 'orange'; ...
    'square' , 'yellow' ; ...
    'square'    , 'green'  ; ...
    'square'  , 'cyan'; ...
    'square'    , 'blue'; ...
    'square'    , 'violet'; ...
    'square', 'brown'};
CONFIG.show_touch_area = 0;
CONFIG.show_map_objs = 0;
CONFIG.repeat_trial_if_error = 0;
CONFIG.main_display = max(Screen('Screens'));
% CONFIG.main_display_area = [0 0 2736 1824] * CONFIG.scale_touchscreen;
CONFIG.n_trials_to_save = 30;
CONFIG.verbose = 0;
% CONFIG.arduino_connected = 0;
CONFIG.time_inbetween_touches = 0.01; % secs
CONFIG.play_sound = 0;
CONFIG.total_trials = 400;
CONFIG.sampleTouchBan = false; % true touch sample is error
CONFIG.pre = true; % true more trials in block, false more trials in random
CONFIG.FlickerPunishment = 2; % flicker punishment time in seconds
CONFIG.shuffle = true;
% CONFIG.repeatFalse = 0; % 0 no repeat; 1 repeat the target only; 2 repeat with the exact distribution of the target and distractor
% CONFIG.block_shrink = 1/4;
% CONFIG.block_num = 20*CONFIG.block_shrink;
if CONFIG.pre
    CONFIG.block_num = 20;
else
    CONFIG.block_num = 10;
end
training_block_cumsum = (1:2*CONFIG.n_pairs)*CONFIG.block_num/2;
test_block_cumsum = repmat(cumsum(0:CONFIG.block_num:CONFIG.n_pairs*CONFIG.block_num),2,1);
test_block_cumsum = test_block_cumsum(:);
rotation_angle = 0;

% Reset the random seed based on time
rng shuffle

for ii=1:CONFIG.n_pairs
    testSeq = repmat(1:2*ii,CONFIG.block_num/2,1);
    testSeqs{ii} = Shuffle(testSeq(:));
end
CONFIG.total_trials = training_block_cumsum(end) + test_block_cumsum(end);
%%% MAIN %%%

if CONFIG.n_distractors < 0 || CONFIG.n_distractors > 5
    error('Invalid number of distractors. It should be between 0 and 5');
end

% To save a copy of the code
if CONFIG.save_results
    % Create the logs dir if necessary
    if ~exist(CONFIG.copy_code_dir,'dir')
        mkdir(CONFIG.copy_code_dir);
    end
%     % Save the main code
%     save_copy_code(CONFIG.copy_code_dir, fname);
%     % Save the config file
%     save_copy_code(sprintf('CONFIG_DM2S_%s', upper(CONFIG.monkey)), sprintf('%s/CONFIG_DM2S.m', fileparts(mfilename('fullpath'))), CONFIG.copy_code_dir, 1);
%     % Filename with date and hour
%     [~,filename] = fileparts(filename);
end

% Detect an Arduino Uno
% if CONFIG.arduino_connected
%     ard = arduino('/dev/ttyACM0', 'Leonardo', 'TraceOn', false);
% end
try
weightPara = load(weightPath);
b = weightPara.b_weight;
k = weightPara.k_weight;
catch
    weightON = false;
end
try
    ard = arduino('/dev/ttyACM0');
    give_juice(ard, 0.1);
%     a = arduino('/dev/ttyACM1');
    CONFIG.arduino_connected = 1;    
catch
    ard = [];
    CONFIG.arduino_connected = 0;
end

if CONFIG.arduino_connected 
    if weightON
        LoadCell = addon( ard, 'basicHX711/basic_HX711',{'D4','D5'} );
        Y_RealTime_HX711 = read_HX711 ( LoadCell );
        weight = ( Y_RealTime_HX711 - b )/ k ;
        fprintf('Weight: %f Kg\n', weight);
    end
    give_juice(ard, 0.1);
end

% Audio handles
PAHANDLES.start_trial = [];
PAHANDLES.success = [];
PAHANDLES.fail = [];

% Setup sounds
if CONFIG.play_sound
    [PAHANDLES] = setup_sound_effects(PAHANDLES, CONFIG);
end

% Setup useful PTB defaults:
PsychDefaultSetup(2);

% % Detect touchscreen
dev = GetTouchDeviceIndices([], 1,  touchscreenID);

% To make it more professional, look at https://www.mathworks.com/help/matlab/matlab_prog/overview-of-the-map-data-structure.html
palette = {'white', [255 255 255] / 2; ...
    'black',  [0 0 0]; ...
    'gray',   [192 192 192] / 255; ...
    'brown',  [165 42  42 ] / 255; ...
    'red',    [255 0   0  ] / 255; ...
    'orange', [255 127 0  ] / 255; ...
    'yellow', [255 255 0  ] / 255; ...
    'green',  [0   255 0  ] / 255; ...
    'cyan',   [0   255 255] / 255; ...
    'blue',   [0   0   255] / 255; ...
    'purple', [75  0   130] / 255; ...
    'violet', [143 0   255] / 255};
black = palette{contains(palette(:,1),'black'),2};
gray = palette{contains(palette(:,1),'gray'),2};
white = palette{contains(palette(:,1),'white'),2};
red = palette{contains(palette(:,1),'red'),2};
% Open display screen
[w, rect] = PsychImaging('OpenWindow', CONFIG.main_display, black, CONFIG.main_display_area);
% [w, rect] = Screen('OpenWindow', CONFIG.main_display, black, CONFIG.main_display_area);
dFigures = dir(fullfile(figfolder,'*.jpg'));
if CONFIG.shuffle
    [dFigures, iShuffle] = Shuffle(dFigures);
else
    iShuffle = 1:length(dFigures);
end
% dd = dFigures(1:12);
% dFigures(1:12) = dFigures(13:24);
% dFigures(13:24) = dd;
sampleIndex = nan(2*CONFIG.n_pairs,1);
% textureIndex = cell(CONFIG.n_distractors+2,1);
% figs = cell(length(dFigures),1);
for ii=1:2*CONFIG.n_pairs
    fig = imread(fullfile(figfolder,dFigures(ii).name));
    sampleIndex(ii) = Screen('MakeTexture', w, fig);
end
% dFigures = dir(fullfile(figfolder,'*.png'));
targetIndex = nan(length(dFigures),1);
for ii=1:length(dFigures)
    fig = imread(fullfile(figfolder,dFigures(ii).name));
    targetIndex(ii) = Screen('MakeTexture', w, fig);
end
% CONFIG.n_pairs = length(dFigures);
% CONFIG.total_trials = 50*CONFIG.n_pairs; 

% Set blend function for alpha blending
Screen('BlendFunction', w, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Size of text displayed in the animation
Screen('TextSize', w, 100 * CONFIG.scale_touchscreen);

% Get maximum supported dot diameter for smooth dots &
% select good diameter for touch point blobs, but no more than what
% 'DrawDots' supports:
[~, maxSmoothPointSize] = Screen('DrawDots', w);
baseSize = min(RectWidth(rect) / 40, maxSmoothPointSize);

% Cell with all possible coords of each object reshaped
% 1 = Onset circle
% 2 = Rect touch
% 3 = Squares
% 4 = Triangles
% 5 = Crosses
% 6 = Octagons
% 7 = Viruses
% 8 = Spaceships
[coords_all_objs, library_objs] = calculate_coords_polygons(CONFIG);

% To add a 3rd column to the objects with the values of the colors
list_objs = CONFIG.list_objs;
[tf, idx] = ismember(list_objs(:,2), palette(:,1));
vals = palette(:,2);
list_objs = [list_objs palette(idx(tf),2)];

% [Optional] To show a map of the squares and square IDs before the program starts
if CONFIG.show_map_objs
    radius_square = floor(CONFIG.half_side_square);
    color_text = red;
    
    % Onset polygon / circle
    Screen('FillPoly', w, color_text, coords_all_objs{1}, 1);
    Screen('Flip', w);
    WaitSecs(0.5);
    
    % Rect touches
    Screen('FillRect', w, white, coords_all_objs{2});
    for ii = 1 : CONFIG.n_distractors + 1
        Screen('DrawText', w, sprintf('%d', ii), coords_all_objs{3}{ii, 1}(1,1) + radius_square - 30 * CONFIG.scale_touchscreen, ...
            coords_all_objs{3}{ii, 1}(1,2) + radius_square - 30 * CONFIG.scale_touchscreen, color_text);
    end
    Screen('Flip', w);
    WaitSecs(0.5);
    
    % Squares
    color = list_objs{contains(list_objs(:,1),'square'),3};
    for ii = 1 : CONFIG.n_distractors + 1
        Screen('FillPoly', w, color, coords_all_objs{3}{ii,1}, 1);
        Screen('DrawText', w, sprintf('%d', ii), coords_all_objs{3}{ii, 1}(1,1) + radius_square - 30 * CONFIG.scale_touchscreen, ...
            coords_all_objs{3}{ii, 1}(1,2) + radius_square - 30 * CONFIG.scale_touchscreen, color_text);
    end
    Screen('Flip', w);
    WaitSecs(0.5);
    
    % Triangles
    color = list_objs{contains(list_objs(:,1),'triangle'),3};
    for ii = 1 : CONFIG.n_distractors + 1
        Screen('FillPoly', w, color, coords_all_objs{4}{ii,1}, 1);
        Screen('DrawText', w, sprintf('%d', ii), coords_all_objs{3}{ii, 1}(1,1) + radius_square - 30 * CONFIG.scale_touchscreen, ...
            coords_all_objs{3}{ii, 1}(1,2) + radius_square - 30 * CONFIG.scale_touchscreen, color_text);
    end
    Screen('Flip', w);
    WaitSecs(0.5);
    
    % Crosses
    color = list_objs{contains(list_objs(:,1),'cross'),3};
    for ii = 1 : CONFIG.n_distractors + 1
        Screen('FillPoly', w, color, coords_all_objs{5}{ii,1});
        Screen('DrawText', w, sprintf('%d', ii), coords_all_objs{3}{ii, 1}(1,1) + radius_square - 30 * CONFIG.scale_touchscreen, ...
            coords_all_objs{3}{ii, 1}(1,2) + radius_square - 30 * CONFIG.scale_touchscreen, color_text);
    end
    Screen('Flip', w);
    WaitSecs(0.5);
    
    % Octagons
    color = list_objs{contains(list_objs(:,1),'octagon'),3};
    for ii = 1 : CONFIG.n_distractors + 1
        Screen('FillPoly', w, color, coords_all_objs{6}{ii,1});
        Screen('DrawText', w, sprintf('%d', ii), coords_all_objs{3}{ii, 1}(1,1) + radius_square - 30 * CONFIG.scale_touchscreen, ...
            coords_all_objs{3}{ii, 1}(1,2) + radius_square - 30 * CONFIG.scale_touchscreen, color_text);
    end
    Screen('Flip', w);
    WaitSecs(0.5);
    
    % Viruses
    color = list_objs{contains(list_objs(:,1),'virus'),3};
    for ii = 1 : CONFIG.n_distractors + 1
        Screen('FillPoly', w, color, coords_all_objs{7}{ii,1});
        Screen('DrawText', w, sprintf('%d', ii), coords_all_objs{3}{ii, 1}(1,1) + radius_square - 30 * CONFIG.scale_touchscreen, ...
            coords_all_objs{3}{ii, 1}(1,2) + radius_square - 30 * CONFIG.scale_touchscreen, color_text);
    end
    Screen('Flip', w);
    WaitSecs(0.5);
    
    % Spaceships
    color = list_objs{contains(list_objs(:,1),'spaceship'),3};
    for ii = 1 : CONFIG.n_distractors + 1
        Screen('FillPoly', w, color, coords_all_objs{8}{ii,1});
        Screen('DrawText', w, sprintf('%d', ii), coords_all_objs{3}{ii, 1}(1,1) + radius_square - 30 * CONFIG.scale_touchscreen, ...
            coords_all_objs{3}{ii, 1}(1,2) + radius_square - 30 * CONFIG.scale_touchscreen, color_text);
    end
    Screen('Flip', w);
    WaitSecs(0.5);
    
    % Pentagons
    color = list_objs{contains(list_objs(:,1),'pentagon'),3};
    for ii = 1 : CONFIG.n_distractors + 1
        Screen('FillPoly', w, color, coords_all_objs{9}{ii,1});
        Screen('DrawText', w, sprintf('%d', ii), coords_all_objs{3}{ii, 1}(1,1) + radius_square - 30 * CONFIG.scale_touchscreen, ...
            coords_all_objs{3}{ii, 1}(1,2) + radius_square - 30 * CONFIG.scale_touchscreen, color_text);
    end
    Screen('Flip', w);
    WaitSecs(0.5);
end

%% Function to process touch events
    function process_touch_event()
        
        % TouchEventAvail reports the number of events in a touch queue.
        % One single touch can contain many events.
        while TouchEventAvail(dev)
            
            evt_count = evt_count + 1;
            
            %  Return oldest pending event
            evt = TouchEventGet(dev, w);
            
            % Touch blob id - Unique in the session at least as
            % long as the finger stays on the screen:
            id = evt.Keycode;
            % fprintf("%d %d\n", evt.Keycode, evt.Type)
            
            % Only consider the id of the firstard = serialport('/dev/ttyACM0', 9600); touch event
            if evt_count == 1
                first_event_id = id;
            end
            
            if id == first_event_id
                switch evt.Type
                    case 0
                        % Not a touch point, but a button press or release on a
                        % physical (or emulated) button associated with the
                        % touch device:ard = serialport('/dev/ttyACM0', 9600);
                        buttonstate = evt.Pressed;
                        
                    case 1
                        % Not really a touch point, but movement of the
                        % simulated mouse cursor, driven by the primary
                        % touch-point:
                        Screen('DrawDots', w, [evt.MappedX; evt.MappedY], ...
                            baseSize, [1,1,1], [], 1, 1);
                        
                    case {2, 3}
                        % 2: New touch point -> New blob!
                        % 3: Moving touch point -> Moving blob!
                        blob.mul = 1.0; % size of the blob
                        blob.x = evt.MappedX;
                        blob.y = evt.MappedY;
                        blob.t = evt.Time;
                        
                    case 4
                        % Touch released -> Dying blob!
                        blob.mul = 0;
                        blob.x = evt.MappedX;
                        blob.y = evt.MappedY;
                        
                    case 5
                        % Lost touch data for some reason:
                        % Flush screen red for one video refresh cycle.
                        fprintf(['Ooops - Sequence data loss! 3rd party ' ...
                            'interference or overload?\n']);
                        Screen('FillRect', w, [1 0 0]);
                        Screen('Flip', w);
                        Screen('FillRect', w, 0);
                end
            end
        end
        
        % Now that all touches for this iteration are processed, repaint
        % the live blob in its new position or fade out a dying blob
        if ~isempty(blob) && blob.mul > 0.1
            % Draw the blob: .mul defines size of the blob:
            Screen('DrawDots', w, [blob.x, blob.y], ...
                blob.mul * baseSize, white, [], 1, 1);
        else
            % Below threshold: Kill the blob
            blob = [];
        end
        
        % To determine if blobcol is empty
        if evt_count && isempty(blob)
            evt_count = 0;
        end
        
        if buttonstate
            Screen('FrameRect', w, [1, 1, 0], [], 5);
        end
        
    end


%%
%% Function to initialize parameters
    function initializePara()
        tclock = clock;
        today = tclock(1:3);
        expBegin = datenum([today 11 0 0]);
        expEnd = datenum([today 14 0 0]);
        nextDay = datenum([today 0 0 0]) + 1;      
        checkPoint = datenum([today 10 0 0]);
        
        if CONFIG.shuffle
            [dFigures, iShuffle] = Shuffle(dFigures);
        else
            iShuffle = 1:length(dFigures);
        end
        % dd = dFigures(1:12);
        % dFigures(1:12) = dFigures(13:24);
        % dFigures(13:24) = dd;
        sampleIndex = nan(2*CONFIG.n_pairs,1);
        % textureIndex = cell(CONFIG.n_distractors+2,1);
        % figs = cell(length(dFigures),1);
        for ii=1:2*CONFIG.n_pairs
            fig = imread(fullfile(figfolder,dFigures(ii).name));
            sampleIndex(ii) = Screen('MakeTexture', w, fig);
        end
        % dFigures = dir(fullfile(figfolder,'*.png'));
        targetIndex = nan(length(dFigures),1);
        for ii=1:length(dFigures)
            fig = imread(fullfile(figfolder,dFigures(ii).name));
            targetIndex(ii) = Screen('MakeTexture', w, fig);
        end

        
        subfolder = datestr(tclock,'yyyymmdd');
        folderpath = fullfile(CONFIG.results_dir,subfolder);
        if ~exist(folderpath,'dir')
            mkdir(folderpath)
        end
        fname = [setupID '-' datestr(tclock,30) '-' monkeyID];
        filename = [fname '.mat'];
        filepath = fullfile(folderpath,filename);
%         filename = [fname '.txt'];
%         txtpath = fullfile(folderpath,filename);
%         fid = fopen(txtpath,'w');

        tdata = nan(10000,4);
        % TRIAL structure
        TRIAL = struct(...  
            'sublist',[], ...
            'sampleID', 0, ...
            'targetID', 0, ... % 1 = Square; 2 = triangle; 3 = cross; 4 = octagon
            'touch_log', [], ... % WRONG FOR THE TIME BEING
            'weight_log',[],...
            'intertrial_period_start', 0, ...
            'onset_period_start', 0, ...
            'pre_sample_period_start', 0, ...
            'sample_period_start', 0, ...
            'pre_test_period_start', 0, ...
            'test_period_start', 0, ...
            'intertrial_period_successful', 0, ...
            'onset_period_successful', 0, ...
            'pre_sample_period_successful', 0, ...
            'sample_period_successful', 0, ...
            'pre_test_period_successful', 0, ...
            'test_period_successful', 0, ...
            'rt_onset', 0, ...
            'rt_test', 0, ...
            'error_type', 0, ...
            'error_msg', '', ...
            'block_type', '' ...
            );
        TRIALS = repmat({TRIAL},1000,1);
        
        total_trials = 0;
        escape_pressed = 0;
        successful_trials = 0;
        consecutive_expired_sessions = 0;
        intertrial_time_punishment = 0;
        TRIALS_filenames = {};
        SESSION.TRIALS = [];
        color_rect_touch = [white 0.3];
        evt_count = 0;
        first_event_id = [];
        buttonstate = 0;
        blackBG = true;
        updatePattern = true;
        sampleID = 1;
        targetID = 2;
    end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                              Task                               %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% HideCursor(w);

% Initializations

expBegin = [];
expEnd = [];
nextDay = [];
checkPoint = [];

total_trials = 0;
escape_pressed = 0;
successful_trials = 0;
consecutive_expired_sessions = 0;
intertrial_time_punishment = 0;
TRIALS_filenames = {};
SESSION.TRIALS = [];
color_rect_touch = [];
evt_count = 0;
first_event_id = [];
buttonstate = 0;
blackBG = true;
folderpath = [];
fname = [];
filepath = [];
TRIALS = [];
updatePattern = 0;

initializePara();

% Create and start touch queue for window and device:
TouchQueueCreate(w, dev);
TouchQueueStart(dev);

% Wait for the go!
KbReleaseWait;

% Only ESCape allows to exit the demo:
RestrictKeysForKbCheck(KbName('ESCAPE'));

% Create a text file to log touch behavior in real time.
if CONFIG.save_results
%     datenow = datestr(now,'mmm-dd-yyyy HH:MM:SS');
%     datePrefix = sprintf('%s_%sh%sm%ss',datenow(1:11),datenow(13:14),datenow(16:17),datenow(19:20));
%     filename = sprintf('%s_%s', datePrefix,mfilename);
%     update_touch_log(CONFIG, filename)
    update_touch_log(CONFIG, fname)
    % Save the main code
    save_copy_code(folderpath, fname);
end

%% Main loop. Press ESC to exit
if CONFIG.verbose
    fprintf('Session started\n');
end

timer_session_start = Screen('Flip', w);
t0 = tic;
while true
    escape_pressed = KbCheck;
    if escape_pressed
        break;
    end
    tnow = now;
    if tnow>nextDay
%         TRIAL.trial_number = total_trials;
%         TRIALS{total_trials} = TRIAL;
%         trials = TRIALS(1:total_trials);
%         save(filepath,'trials','timer_session_start','iShuffle');
        initializePara;
        update_touch_log(CONFIG, fname)
       
        % Save the main code
        save_copy_code(folderpath, fname);
        timer_session_start = Screen('Flip', w);
    end
    
    if tnow>expEnd && total_trials>0
%         break;
        TRIAL.trial_number = total_trials;
        TRIALS{total_trials} = TRIAL;
        trials = TRIALS(1:total_trials);
        save(filepath,'trials','timer_session_start','iShuffle');
    end
    if tnow<expBegin || tnow>expEnd
        if ~blackBG
            Screen('FillRect', w, black);
            Screen('Flip', w);
            blackBG = true;
        end
        if tnow>checkPoint && tnow<expBegin
            pause(1);
        else
            pause(60);
        end
        continue;
    end
    
    if blackBG
        Screen('Flip', w);
        blackBG = false;
        TouchEventFlush(dev);
    end
    
    if toc(t0)>60 % Check whether the program is stuck
        fprintf('%s\n',datestr(now))
        t0 = tic;
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%                  Intertrial Interval                    %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if CONFIG.verbose
        fprintf('-----------------------------------------------------\n');
        fprintf('Intertrial period started\n');
    end
    
    % It's important to reinitialize at each period because the reaction time is discounted from the event. If the touch isn't moved or released, rt will be negative
    evt = [];
    blob = [];
    
    % A variable used in case the monkey keeps touching the screen
    monkey_touching = 0;
    
    % 0: intertrial, 1: onset, 2: pre-sample, 3: sample: 4: pre-test, 5: test
    task_period = 0;
    
    if total_trials
        
        TRIAL.trial_number = total_trials;
        
        % Add the last trial to the end of TRIALS
%         TRIALS(mod(total_trials - 1, CONFIG.n_trials_to_save) + 1) = TRIAL;
        TRIALS{total_trials} = TRIAL;
        
        % Save temporary TRIALS every 'CONFIG.n_trials_to_save' trials. This is useful because if the program crashes, the majority of the data will be saved
        if ~mod(total_trials, CONFIG.n_trials_to_save) && CONFIG.save_results
            %             TRIALS_filenames = vertcat(TRIALS_filenames, {sprintf('%s/%s_trials_%.4d_to_%.4d.mat', ...
            %                 CONFIG.results_dir, filename, total_trials - CONFIG.n_trials_to_save + 1, total_trials)});
            %                 save(TRIALS_filenames{end,:}, 'TRIALS');
            trials = TRIALS(1:total_trials);
            save(filepath,'trials','timer_session_start','iShuffle');
        end
        
        % Update behavior log
        if CONFIG.save_results
            update_touch_log (CONFIG, fname, TRIAL);
        end
    end

    if total_trials >= CONFIG.total_trials
        expEnd = now;
        continue;
    end
    
    % TRIAL structure
    TRIAL = TRIALS{end};
    TRIAL.block_type = CONFIG.n_distractors;
    
    % Determine the position of the objects each block
    if (~exist('TRIAL','var') || ~CONFIG.repeat_trial_if_error || (CONFIG.repeat_trial_if_error && ~TRIAL.error_type))
        
%         if ~CONFIG.n_distractors||updatePattern
%             [coords_all_objs] = calculate_coords_polygons(CONFIG);
%             updatePattern = false;
%         end
        
        if mod(targetID,2)==0
            sample_id = targetID - 1;
        else
            sample_id = targetID + 1;
        end

        % To select a sublist of objects to use in this trial
        sublist_objs_idx = randperm(length(dFigures)-1, CONFIG.n_distractors + 1);
        target_list = setdiff(1:length(dFigures),sample_id); 
        sublist_objs_idx = target_list(sublist_objs_idx);
        
        % To randomly select the target
        target_pos = randperm(CONFIG.n_distractors + 1, 1);

        if sublist_objs_idx(target_pos)~=targetID%total_trials<2*CONFIG.n_pairs*CONFIG.block_num
            sublist_objs_idx(sublist_objs_idx==targetID) = sublist_objs_idx(target_pos);
            sublist_objs_idx(target_pos) = targetID;
        else
            
        end
        
        if ~CONFIG.n_distractors||updatePattern
            [coords_all_objs] = calculate_coords_polygons(CONFIG);
            updatePattern = false;
        end
        TRIAL.sublist = sublist_objs_idx;
        TRIAL.sampleID = sample_id;
        TRIAL.targetID = sublist_objs_idx(target_pos);
        
%         sublist_objs = list_objs(sublist_objs_idx,:);
%         target_type = sublist_objs{target_pos, 1};
%         target_type_test = sublist_objs{end, 1};
%         Screen('Close', textureIndex);
%         for ii=1:CONFIG.n_distractors
%             textureIndex{ii} = Screen('MakeTexture', w, figs{12+sublist_objs_idx(ii)});
%         end
%         textureIndex{end} = Screen('MakeTexture', w, figs{TRIAL.targetID});
        
%         [~, idx] = ismember(target_type, library_objs);
%         target_coords_centered = coords_all_objs{idx}{CONFIG.n_distractors + 2};
        target_coords_centered = coords_all_objs{2}(:,CONFIG.n_distractors + 2);
        
        % To find the coordinates of the objects in the sublist
        % library_objs = {'circle', 'rect', 'square', 'triangle', 'cross', 'octagon', 'virus', 'spaceship'};
%         [~, idx] = ismember(sublist_objs(:,1), library_objs);
%         coords_objs_sample = cell(1, CONFIG.n_distractors + 1);
%         color_objs_sample = sublist_objs(1:end-1,3);
%         for ii = 1 : CONFIG.n_distractors + 1
%             coords_objs_sample{ii} = coords_all_objs{idx(ii)}{ii};
%         end
%         coords_objs_test = coords_objs_sample;
        coords_objs_test = coords_all_objs{2}(:,1 : CONFIG.n_distractors + 1);
%         color_objs_test = color_objs_sample;
%         [~, idx] = ismember(target_type_test, library_objs);
%         coords_objs_test{target_pos} = coords_all_objs{idx}{target_pos};
%         color_objs_test{target_pos} = sublist_objs{end,3};
    end
    
%     if CONFIG.verbose
%         fprintf('\nTarget: %s, pos. %d\n', target_type, target_pos);
%         for ii = 1 : CONFIG.n_distractors + 1
%             fprintf('Pos. %d: %s\n', ii, sublist_objs{ii,1});
%         end
%         fprintf('\n');
%     end    
    
    % [Workaround for a bug in psychtoolbox] To preallocate the structure TRIALS every 30 trials
    if ~mod(total_trials, CONFIG.n_trials_to_save)
%         TRIALS(1:CONFIG.n_trials_to_save) = TRIAL;
        TouchQueueStop(dev);
        TouchQueueRelease(dev);
        pause(0.2);
        TouchQueueCreate(w, dev);
        TouchQueueStart(dev);
    end
    
    % Flip screen
    timer = Screen('Flip', w);
    
    % Timer for intertrial interval
    timer_end = timer + CONFIG.intertrial_interval + intertrial_time_punishment;
    intertrial_time_punishment = 0;
    
    % Record start of sample period (relative to session start)
    TRIAL.intertrial_period_start = timer - timer_session_start;
    
    % Record start of sample period (absolute)
    intertrial_period_start_abs = timer;
    
    % Timer between the recordings of 2 touches
    timer_inbetween_touches = GetSecs - CONFIG.time_inbetween_touches;
    
    while ~TRIAL.error_type && ~TRIAL.intertrial_period_successful && ~escape_pressed
        
        % Escape stim
        escape_pressed = KbCheck;
        
        % Record touches. Touches must have 'CONFIG.time_inbetween_touches' interval between them
        if ~isempty(blob) && (evt.Type == 2 || evt.Type == 3) && ...
                GetSecs >= intertrial_period_start_abs + CONFIG.time_inbetween_touches
            % Timer between the test and intertrial. Otherwise, sometimes a touch
            % inside the test period will be also recorded in the intertrial period.
            
            % Timer inbetween touches
            if GetSecs >= timer_inbetween_touches + CONFIG.time_inbetween_touches
                
                timer_inbetween_touches =  GetSecs;
                
                % Touch details
                TRIAL.touch_log(end + 1, 13) = 0; % To initialize all the 13 cols with 0
                TRIAL.touch_log(end, 1) = total_trials + 1;
                TRIAL.touch_log(end, 2) = task_period;
                TRIAL.touch_log(end, 3) = evt.MappedX;
%                 TRIAL.touch_log(end, 4) = CONFIG.main_display_area(4) - evt.MappedY;
                TRIAL.touch_log(end, 4) = evt.MappedY;
                TRIAL.touch_log(end, 11) = evt.Type;
            end
        end
        
        process_touch_event;
        timer = Screen('Flip', w);
        
        % Timer expired
        if timer > timer_end
            TRIAL.intertrial_period_successful = 1;
        end
    end
    
    % Don't start trial if the monkey keeps touching the screen and restart it with untouch
    Screen('Flip', w); % Necessary to clean the blob while keeping touch
    while ~isempty(evt) && (evt.Type == 2 || evt.Type == 3) && ~escape_pressed
        monkey_touching = 1;
        escape_pressed = KbCheck;
        process_touch_event;
    end
    % [Workaround] Create a black window on top of the path of the white dot
    Screen('FillRect', w, gray, rect);
    % To restart the intertrial interval if the monkey kept touching the screen
    if monkey_touching
        continue;
    end
    
    % End session
    if escape_pressed
        break;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%                         ONSET                           %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    total_trials =  total_trials + 1;
    
    if CONFIG.verbose
        fprintf('\nTRIAL: %d\n', total_trials);
        fprintf('Onset\n');
    end
    
    % 0: intertrial, 1: onset, 2: pre-sample, 3: sample: 4: pre-test, 5: test
    task_period = 1;
    
    % Onset circle in the middle of the screen
    Screen('FillPoly', w, red, coords_all_objs{1}, 1); % onset polygon / circle in the middle
    
    if CONFIG.show_touch_area
        Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(1:4,end)); % rect touch
    end
    
    % Screen flip
    timer = Screen('Flip', w);
    
    % Time limit to touch the square
    timer_end = timer + CONFIG.max_time_touch_onset;
    
    % Store start of this trial relative to session start
    TRIAL.onset_period_start = timer - timer_session_start;
    
    % Store start of this trial (absolute time)
    onset_period_start_abs = timer;
    
    % Timer between the recordings of 2 touches
    timer_inbetween_touches = GetSecs - CONFIG.time_inbetween_touches;
    
    if CONFIG.arduino_connected && weightON
        Y_RealTime_HX711 = read_HX711 ( LoadCell );
        weight = ( Y_RealTime_HX711 - b )/ k ;
        TRIAL.weight_log(end+1,:) = [weight timer - timer_session_start];
        fprintf('Weight: %f Kg\n', weight);
        lastMeasure = timer;
    end
    
    % while ~TRIAL.error_type && ~TRIAL.test_period_successful && ~escape_pressed
    while ~TRIAL.error_type && ~TRIAL.onset_period_successful && tnow<expEnd
        tnow = now;
        % Escape stim
        escape_pressed = KbCheck;
        
        % Draw randomly selected objects
        % Draw all the objects
        Screen('FillPoly', w, red, coords_all_objs{1}, 1); % onset polygon / circle in the middle
        if CONFIG.show_touch_area
            Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(1:4,end)); % rect touch
        end
        
        % Store touches. Touches must have 'CONFIG.time_inbetween_touches' s interval between them
        if ~isempty(blob) && ~isempty(evt) && (evt.Type == 2 || evt.Type == 3) && GetSecs >= onset_period_start_abs + CONFIG.time_inbetween_touches
            
            % Timer inbetween touches
            if GetSecs >= timer_inbetween_touches + CONFIG.time_inbetween_touches
                
                timer_inbetween_touches =  GetSecs;
                
                % Touch details
                TRIAL.touch_log(end + 1, 13) = 0;
                TRIAL.touch_log(end, 1) = total_trials;
                TRIAL.touch_log(end, 2) = task_period;
                TRIAL.touch_log(end, 3) = evt.MappedX;
                TRIAL.touch_log(end, 4) = evt.MappedY;
                TRIAL.touch_log(end, 6) = onset_period_start_abs - intertrial_period_start_abs;
                TRIAL.touch_log(end, 9) = evt.Time - onset_period_start_abs;
                TRIAL.touch_log(end, 11) = evt.Type;
                
                % Detect whether the blob is inside the onset polygon / circle
                % blob_inside_target = IsInRect(blob.x, blob.y, reshape(coords_onset_pol,[2, size(coords_onset_pol,2) / 2])');
%                 blob_inside_target = IsInRect(blob.x, blob.y, CenterRect(coords_all_objs{2}(1:4,end), rect));
                blob_inside_target = IsInRect(blob.x, blob.y, coords_all_objs{2}(1:4,end));
                
                % Monkey touched
                if evt_count > 0 && blob_inside_target
                    TRIAL.touch_log(end, 13) = 1; % To record a success
                    TRIAL.rt_onset = evt.Time - onset_period_start_abs;
                    TRIAL.onset_period_successful = 1;
                    % Play sound
                    if CONFIG.play_sound
                        play_sound_effect(PAHANDLES.success. CONFIG);
                    end
                    % Flickering
                    for i_flicker = 1 : 6
                        Screen('Flip', w);
                        if CONFIG.show_touch_area
                            Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(1:4,end)); % rect touch
                        end
                        Screen('Flip', w);
                        if CONFIG.show_touch_area
                            Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(1:4,end)); % rect touch
                        end
                        Screen('Flip', w)
                        Screen('FillPoly', w, red, coords_all_objs{1}, 1); % onset polygon / circle in the middle
                        if CONFIG.show_touch_area
                            Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(1:4,end)); % rect touch
                        end
                    end
                    
                    % Break if the blob (touch) happens outside the target
                elseif evt_count > 0 && ~blob_inside_target
                    if CONFIG.play_sound
                        play_sound_effect(PAHANDLES.fail. CONFIG);
                    end
                    flicker_screen_punishment(w, CONFIG);
                    TRIAL.error_type = 2;
                    TRIAL.error_msg = 'touch outside target';
                    TRIAL.touch_log(end, 11) = -1;
                    intertrial_time_punishment = CONFIG.intertrial_time_punishment;
%                     fprintf('Touch ouside target!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                end
                
                % While the monkey keeps the blob inside the square, nothing will happen
                while (evt.Type == 2 || evt.Type == 3) && ~escape_pressed && blob_inside_target
                    escape_pressed = KbCheck;
                    Screen('Flip', w);
                    Screen('FillPoly', w, red, coords_all_objs{1}, 1); % onset polygon / circle in the middle
                    if CONFIG.show_touch_area
                        Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(1:4,end)); % rect touch
                    end
                    process_touch_event;
                    if ~isempty(blob)
                        % blob_inside_target = IsInRect(blob.x, blob.y, reshape(coords_onset_pol,[2, size(coords_onset_pol,2) / 2])');
%                         blob_inside_target = IsInRect(blob.x, blob.y, CenterRect(coords_all_objs{2}(1:4), rect));  % rect touch
                        blob_inside_target = IsInRect(blob.x, blob.y, coords_all_objs{2}(1:4,end));
                    end
                end
                
                % If the monkey untouches the screen and the blob is inside the target, it's a success
                if evt.Type == 4 && blob_inside_target
                    break;
                elseif ~blob_inside_target
                    % Break if untouch happens outside the target
                    if TRIAL.error_type == 0
                        if CONFIG.play_sound
                            play_sound_effect(PAHANDLES.fail. CONFIG);
                        end                      
                        flicker_screen_punishment(w, CONFIG);
                        TRIAL.error_type = 3; %%%%% CHANGE HERE
                        TRIAL.error_msg = 'untouch outside target';
                        TRIAL.touch_log(end, 11) = -1;
                        intertrial_time_punishment = CONFIG.intertrial_time_punishment;
%                         fprintf('Untouch outside target!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                    end
                end
                
            end
        end
        
%         % Break if timer expired
%         if timer > timer_end
%             disp('Timer expired');
%             TRIAL.error_type = 1;
%             % Just to flicker the screen if the monkeys doesn't engage in 5 trials
%             consecutive_expired_sessions = consecutive_expired_sessions + 1;
%             if consecutive_expired_sessions == 5
%                 if CONFIG.play_sound
%                     play_sound_effect(PAHANDLES.fail. CONFIG);
%                 end
%                 flicker_screen_punishment(w, CONFIG);
%                 consecutive_expired_sessions = 0;
%             end
%         end
        
        process_touch_event;
        timer = Screen('Flip', w);
        
        if CONFIG.arduino_connected && weightON && timer - lastMeasure > 5
            Y_RealTime_HX711 = read_HX711 ( LoadCell );
            weight = ( Y_RealTime_HX711 - b )/ k ;
            TRIAL.weight_log(end+1,:) = [weight timer - timer_session_start];
            fprintf('Weight: %f Kg\n', weight);
            lastMeasure = timer;
        end
    
        % End session (break from internal while)
        if escape_pressed
            break;
        end
        
    end
    
    % End session (break from external while)
    if escape_pressed
        break;
    end
    
    % To break the trial in case there was an error
    if TRIAL.error_type || tnow>=expEnd
        continue;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%                   INTERVAL PRE-SAMPLE                   %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if CONFIG.verbose
        fprintf('Interval pre-sample\n');
    end
    
    % 0: intertrial, 1: onset, 2: pre-sample, 3: sample: 4: pre-test, 5: test
    task_period = 2;
    
    % Screen flip
    timer = Screen('Flip', w);
    
    % Pre-sample time
    timer_end = timer + CONFIG.pre_sample_period;
    
    % Store start of this trial relative to session start
    TRIAL.pre_sample_period_start = timer - timer_session_start;
    
    % Store start of this trial (absolute time)
    pre_sample_period_start_abs = timer;
    
    % Timer between the recordings of 2 touches
    timer_inbetween_touches = GetSecs - CONFIG.time_inbetween_touches;
    
    % while ~TRIAL.error_type && ~TRIAL.test_period_successful && ~escape_pressed
    while ~TRIAL.error_type && ~TRIAL.pre_sample_period_successful
        % Escape stim
        escape_pressed = KbCheck;
        
        % Store touches. Touches must have 'CONFIG.time_inbetween_touches' s interval between them
        if ~isempty(blob) && ~isempty(evt) && (evt.Type == 2 || evt.Type == 3) && GetSecs >= pre_sample_period_start_abs + CONFIG.time_inbetween_touches
            
            % Timer inbetween touches
            if GetSecs >= timer_inbetween_touches + CONFIG.time_inbetween_touches
                
                timer_inbetween_touches =  GetSecs;
                
                % Touch details
                TRIAL.touch_log(end + 1, 13) = 0;
                TRIAL.touch_log(end, 1) = total_trials;
                TRIAL.touch_log(end, 2) = task_period;
                TRIAL.touch_log(end, 3) = evt.MappedX;
                TRIAL.touch_log(end, 4) = evt.MappedY;
                TRIAL.touch_log(end, 6) = pre_sample_period_start_abs - intertrial_period_start_abs;
                TRIAL.touch_log(end, 9) = evt.Time - pre_sample_period_start_abs;
                TRIAL.touch_log(end, 11) = evt.Type;
                
                if evt_count > 0
                    if CONFIG.play_sound
                        play_sound_effect(PAHANDLES.fail. CONFIG);
                    end
                    flicker_screen_punishment(w, CONFIG);
                    TRIAL.error_type = 2;
                    TRIAL.error_msg = 'touch outside target';
                    TRIAL.touch_log(end, 11) = -1;
                    intertrial_time_punishment = CONFIG.intertrial_time_punishment;
%                     fprintf('Touch outside target!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                end
                
            end
        end
        
        process_touch_event;
        timer = Screen('Flip', w);
        
        % End session (break from internal while)
        if escape_pressed
            break;
        end
        if timer > timer_end
            TRIAL.pre_sample_period_successful = 1;
        end
    end
    
    % End session (break from external while)
    if escape_pressed
        break;
    end
    
    % To break the trial in case there was an error
    if TRIAL.error_type
        continue;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%                         SAMPLE                          %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if CONFIG.verbose
        fprintf('Sample period\n');
    end
    
    % 0: intertrial, 1: onset, 2: pre-sample, 3: sample: 4: pre-test, 5: test
    task_period = 3;
    
    % Show sample
%     Screen('FillPoly', w, list_objs{contains(list_objs(:,1), target_type), 3}, target_coords_centered);
    Screen('DrawTexture', w, sampleIndex(TRIAL.sampleID),[],target_coords_centered,rotation_angle);
%     % Draw all the sample objects
%     for i = 1 : CONFIG.n_distractors + 1
%         Screen('FillPoly', w, color_objs_sample{i}, coords_objs_sample{i});
%     end
    
    % Screen flip
    timer = Screen('Flip', w);
    
    % Time limit to touch the square
    timer_end = timer + CONFIG.sample_period;
    
    % Store start of this trial relative to session start
    TRIAL.sample_period_start = timer - timer_session_start;
    
    % Store start of this trial (absolute time)
    sample_period_start_abs = timer;
    
    % Timer between the recordings of 2 touches
    timer_inbetween_touches = GetSecs - CONFIG.time_inbetween_touches;
    
    % while ~TRIAL.error_type && ~TRIAL.test_period_successful && ~escape_pressed
    while ~TRIAL.error_type && ~TRIAL.sample_period_successful
        
        % Escape stim
        escape_pressed = KbCheck;
        
        % Draw randomly selected objects
        % Draw all the objects
        %         Screen('FillPoly', w, list_objs{contains(list_objs(:,1), target_type), 3}, target_coords_centered);
        Screen('DrawTexture', w, sampleIndex(TRIAL.sampleID),[],target_coords_centered,rotation_angle);
%         for i = 1 : CONFIG.n_distractors + 1
%             Screen('FillPoly', w, color_objs_sample{i}, coords_objs_sample{i});
%         end
        
        % Store touches. Touches must have 'CONFIG.time_inbetween_touches' s interval between them
        if ~isempty(blob) && ~isempty(evt) && (evt.Type == 2 || evt.Type == 3) && GetSecs >= sample_period_start_abs + CONFIG.time_inbetween_touches
            
            % Timer inbetween touches
            if GetSecs >= timer_inbetween_touches + CONFIG.time_inbetween_touches
                
                timer_inbetween_touches =  GetSecs;
                
                % Touch details
                TRIAL.touch_log(end + 1, 13) = 0;
                TRIAL.touch_log(end, 1) = total_trials;
                TRIAL.touch_log(end, 2) = task_period;
                TRIAL.touch_log(end, 3) = evt.MappedX;
                TRIAL.touch_log(end, 4) = evt.MappedY;
                TRIAL.touch_log(end, 6) = sample_period_start_abs - intertrial_period_start_abs;
                TRIAL.touch_log(end, 9) = evt.Time - sample_period_start_abs;
                TRIAL.touch_log(end, 11) = evt.Type;
                                
                blob_inside_sample = 0;
                for ii=1:CONFIG.n_distractors + 1
                    blob_inside_sample = IsInRect(blob.x, blob.y, coords_all_objs{2}(:,ii));
                    if blob_inside_sample
                        break;
                    end
                end
                if evt_count > 0 && (CONFIG.sampleTouchBan || ~blob_inside_sample)
                    if CONFIG.play_sound
                        play_sound_effect(PAHANDLES.fail. CONFIG);
                    end
                    flicker_screen_punishment(w, CONFIG);
%                     blob_inside_target = IsInRect(blob.x, blob.y, CenterRect(coords_all_objs{2}(1:4), rect));  % rect touch
                    
                    if blob_inside_sample
                        TRIAL.error_type = 4;
                        TRIAL.error_msg = 'touch sample in sample period';
%                         fprintf('Touch target in sample period!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                    else
                        TRIAL.error_type = 2;
                        TRIAL.error_msg = 'touch outside target';
%                         fprintf('Touch outside target!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                    end
                    TRIAL.touch_log(end, 11) = -1;
                    intertrial_time_punishment = CONFIG.intertrial_time_punishment;
                end
                
            end
        end
        
        process_touch_event;
        timer = Screen('Flip', w);
        
        % End session (break from internal while)
        if escape_pressed
            break;
        end
        if timer > timer_end
            TRIAL.sample_period_successful = 1;
        end
    end
    
    % End session (break from external while)
    if escape_pressed
        break;
    end
    
    % To break the trial in case there was an error
    if TRIAL.error_type
        continue;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%                    INTERVAL PRE-TEST                    %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if CONFIG.verbose
        fprintf('Interval pre-test\n');
    end
    
    % 0: intertrial, 1: onset, 2: pre-sample, 3: sample: 4: pre-test, 5: test
    task_period = 4;
    
    % Screen flip
    timer = Screen('Flip', w);
    timer_end = timer + CONFIG.pre_test_period;
    
    % Store start of this trial relative to session start
    TRIAL.pre_test_period_start = timer - timer_session_start;
    
    % Store start of this trial (absolute time)
    pre_test_period_start_abs = timer;
    
    % Timer between the recordings of 2 touches
    timer_inbetween_touches = GetSecs - CONFIG.time_inbetween_touches;
    
    % while ~TRIAL.error_type && ~TRIAL.test_period_successful && ~escape_pressed
    while ~TRIAL.error_type && ~TRIAL.pre_test_period_successful
        
        % Escape stim
        escape_pressed = KbCheck;
        
        % Store touches. Touches must have 'CONFIG.time_inbetween_touches' s interval between them
        if ~isempty(blob) && ~isempty(evt) && (evt.Type == 2 || evt.Type == 3) && GetSecs >= pre_test_period_start_abs + CONFIG.time_inbetween_touches
            
            % Timer inbetween touches
            if GetSecs >= timer_inbetween_touches + CONFIG.time_inbetween_touches
                
                timer_inbetween_touches =  GetSecs;
                
                % Touch details
                TRIAL.touch_log(end + 1, 13) = 0;
                TRIAL.touch_log(end, 1) = total_trials;
                TRIAL.touch_log(end, 2) = task_period;
                TRIAL.touch_log(end, 3) = evt.MappedX;
                TRIAL.touch_log(end, 4) = evt.MappedY;
                TRIAL.touch_log(end, 6) = pre_test_period_start_abs - intertrial_period_start_abs;
                TRIAL.touch_log(end, 9) = evt.Time - pre_test_period_start_abs;
                TRIAL.touch_log(end, 11) = evt.Type;
                
                if evt_count > 0
                    if CONFIG.play_sound
                        play_sound_effect(PAHANDLES.fail. CONFIG);
                    end
                    flicker_screen_punishment(w, CONFIG);
                    TRIAL.error_type = 2;
                    TRIAL.error_msg = 'touch outside target';
                    TRIAL.touch_log(end, 11) = -1;
                    intertrial_time_punishment = CONFIG.intertrial_time_punishment;
%                     fprintf('Touch outside target!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                end
                
            end
        end
        
        process_touch_event;
        timer = Screen('Flip', w);
        
        % End session (break from internal while)
        if escape_pressed
            break;
        end
        if timer > timer_end
            TRIAL.pre_test_period_successful = 1;
        end
    end
    
    if CONFIG.arduino_connected && weightON
        Y_RealTime_HX711 = read_HX711 ( LoadCell );
        weight = ( Y_RealTime_HX711 - b )/ k ;
        TRIAL.weight_log(end+1,:) = [weight timer - timer_session_start];
        fprintf('Weight: %f Kg\n', weight);
    end
    
    % End session (break from external while)
    if escape_pressed
        break;
    end
    
    % To break the trial in case there was an error
    if TRIAL.error_type
        continue;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%                          TEST                           %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if CONFIG.verbose
        fprintf('Test\n');
    end
    
    % 0: intertrial, 1: onset, 2: pre-sample, 3: sample: 4: pre-test, 5: test
    task_period = 5;
    
    % Draw all the objects
    for ii = 1 : CONFIG.n_distractors + 1
%         Screen('FillPoly', w, color_objs_test{i}, coords_objs_test{i});
        Screen('DrawTexture', w, targetIndex(sublist_objs_idx(ii)),[],coords_objs_test(:,ii),rotation_angle);
    end
    if CONFIG.show_touch_area
        Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(:,target_pos)); % rect touch
    end
    
%     cell2mat(color_objs_test)
%     cell2mat(coords_objs_test)
    % Screen flip
    timer = Screen('Flip', w);
    
    % Time limit to touch the square
    timer_end = timer + CONFIG.max_time_touch_test;
    
    % Store start of this trial relative to session start
    TRIAL.test_period_start = timer - timer_session_start;
    
    % Store start of this trial (absolute time)
    test_period_start_abs = timer;
    
    % Timer between the recordings of 2 touches
    timer_inbetween_touches = GetSecs - CONFIG.time_inbetween_touches;
    
    %     while ~TRIAL.error_type && ~TRIAL.test_period_successful && ~escape_pressed
    while ~TRIAL.error_type && ~TRIAL.test_period_successful
        
        % Escape stim
        escape_pressed = KbCheck;
        
        % Draw all the objects
        for ii = 1 : CONFIG.n_distractors + 1
%             Screen('FillPoly', w, color_objs_test{i}, coords_objs_test{i});
            Screen('DrawTexture', w, targetIndex(sublist_objs_idx(ii)),[],coords_objs_test(:,ii),rotation_angle);
        end
        if CONFIG.show_touch_area
            Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(:,target_pos)); % rect touch
        end
        
        % Store touches. Touches must have 'CONFIG.time_inbetween_touches' s interval between them
        if ~isempty(blob) && ~isempty(evt) && (evt.Type == 2 || evt.Type == 3) && GetSecs >= test_period_start_abs + CONFIG.time_inbetween_touches
            
            % Timer inbetween touches
            if GetSecs >= timer_inbetween_touches + CONFIG.time_inbetween_touches
                
                timer_inbetween_touches =  GetSecs;
                
                % Touch details
                TRIAL.touch_log(end + 1, 13) = 0;
                TRIAL.touch_log(end, 1) = total_trials;
                TRIAL.touch_log(end, 2) = task_period;
                TRIAL.touch_log(end, 3) = evt.MappedX;
                TRIAL.touch_log(end, 4) = evt.MappedY;
                TRIAL.touch_log(end, 6) = test_period_start_abs - intertrial_period_start_abs;
                TRIAL.touch_log(end, 9) = evt.Time - test_period_start_abs;
                TRIAL.touch_log(end, 11) = evt.Type;
                
                % Detect whether the blob is inside the square
                % blob_inside_target = IsInRect(blob.x, blob.y, coords_target_this_trial);
                blob_inside_target = IsInRect(blob.x, blob.y, coords_all_objs{2}(:,target_pos));
                
                % Monkey touched screen
                if evt_count > 0 && blob_inside_target
                    % Update id of object touched
                    %TRIAL.touch_log(end, 12) = ceil(visible_squares_coords(end, 6) / 2);
                    TRIAL.touch_log(end, 13) = 1; % To record a success
                    TRIAL.rt_test = evt.Time - test_period_start_abs;
                    TRIAL.test_period_successful = 1;
                    if CONFIG.play_sound
                        play_sound_effect(PAHANDLES.success. CONFIG);
                    end
                    % Flickering
                    for i_flicker = 1 : 6
                        Screen('Flip', w);
                        for ii = 1 : CONFIG.n_distractors + 1
                            if ii ~= target_pos
%                                 Screen('FillPoly', w, color_objs_test{i}, coords_objs_test{i});
                                Screen('DrawTexture', w, targetIndex(sublist_objs_idx(ii)),[],coords_objs_test(:,ii),rotation_angle);
                            end
                        end
                        if CONFIG.show_touch_area
                            Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(:,target_pos)); % rect touch
                        end
                        Screen('Flip', w);
                        for ii = 1 : CONFIG.n_distractors + 1
                            if ii ~= target_pos
%                                 Screen('FillPoly', w, color_objs_test{ii}, coords_objs_test{ii});
                                Screen('DrawTexture', w, targetIndex(sublist_objs_idx(ii)),[],coords_objs_test(:,ii),rotation_angle);
                            end
                        end
                        if CONFIG.show_touch_area
                            Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(:,target_pos)); % rect touch
                        end
                        Screen('Flip', w)
                        for ii = 1 : CONFIG.n_distractors + 1
%                             Screen('FillPoly', w, color_objs_test{ii}, coords_objs_test{ii});
                            Screen('DrawTexture', w, targetIndex(sublist_objs_idx(ii)),[],coords_objs_test(:,ii),rotation_angle);
                        end
                        if CONFIG.show_touch_area
                            Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(:,target_pos)); % rect touch
                        end
                    end
                    Screen('Flip', w);
                    
                    % Break if the blob (touch) happens outside the target
                elseif evt_count > 0 && ~blob_inside_target
                    if CONFIG.play_sound
                        play_sound_effect(PAHANDLES.fail. CONFIG);
                    end
                    flicker_screen_punishment(w, CONFIG);
                    blob_inside_wrong_target = 0;
                    for ii=1:CONFIG.n_distractors + 1
                        blob_inside_wrong_target = IsInRect(blob.x, blob.y, coords_all_objs{2}(:,ii));
                        if blob_inside_wrong_target
                            break;
                        end
                    end
                    if blob_inside_wrong_target
                        TRIAL.error_type = 5;
                        TRIAL.error_msg = 'touch inside wrong target';
%                         fprintf('Touch inside wrong target!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                    else
                        TRIAL.error_type = 2;
                        TRIAL.error_msg = 'touch outside target';
%                         fprintf('Touch outside target!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                    end
                    TRIAL.touch_log(end, 11) = -1;
                    intertrial_time_punishment = CONFIG.intertrial_time_punishment;
                end
                
                % While the monkey keeps the blob inside the square, nothing will happen
                while (evt.Type == 2 || evt.Type == 3) && ~escape_pressed && blob_inside_target
                    escape_pressed = KbCheck;
                    Screen('Flip', w); % Necessary to clean the blob while keeping touch
                    for ii = 1 : CONFIG.n_distractors + 1
%                         Screen('FillPoly', w, color_objs_test{ii}, coords_objs_test{ii});
                        Screen('DrawTexture', w, targetIndex(sublist_objs_idx(ii)),[],coords_objs_test(:,ii),rotation_angle);
                    end
                    if CONFIG.show_touch_area
                        Screen('FillRect', w, color_rect_touch, coords_all_objs{2}(:,target_pos)); % rect touch
                    end
                    process_touch_event;
                    
                    if ~isempty(blob)
                        blob_inside_target = IsInRect(blob.x, blob.y, coords_all_objs{2}(:,target_pos));
                    end
                end
                
                % If the monkey untouches the screen and the blob is inside the target, it's a success
                if evt.Type == 4 && blob_inside_target
                    % Reward
                    if CONFIG.arduino_connected
                        % To give juice using the arduino
                        give_juice(ard, water_time);
                    else
                        Screen('DrawText', w, 'Correct!', CONFIG.main_display_area(3)/2 - 180 * CONFIG.scale_touchscreen, CONFIG.main_display_area(4)/2 - 250 * CONFIG.scale_touchscreen - CONFIG.half_side_square, red);
                        Screen('Flip', w);
                    end
                    
                    % Successes
                    successful_trials = successful_trials + 1;
                    
                    
                    if 1%CONFIG.verbose
                        fprintf('Trial successful!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                        fprintf('Reaction_time: %f\n', TRIAL.touch_log(end, 9));
                    end
                    
                elseif ~blob_inside_target
                    % Break if untouch happens outside the target
                    if TRIAL.error_type == 0
                        if CONFIG.play_sound
                            play_sound_effect(PAHANDLES.fail. CONFIG);
                        end
                        flicker_screen_punishment(w, CONFIG);
                        TRIAL.error_type = 3; %%%%% CHANGE HERE
                        TRIAL.error_msg = 'untouch outside target';
                        TRIAL.touch_log(end, 11) = -1;
                        intertrial_time_punishment = CONFIG.intertrial_time_punishment;
%                         fprintf('Untouch outside target!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
                    end
                end
                
            end
        end
        
        % Break if timer expired
        if timer > timer_end
            disp('Timer expired');
            TRIAL.error_type = 1;
            TRIAL.error_msg = 'time expired';
            TRIAL.touch_log(end, 11) = -1;
%             fprintf('Time out!\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / (total_trials));
        end
        
        process_touch_event;
        timer = Screen('Flip', w);
        
        % End session (break from internal while)
        if escape_pressed
            break;
        end
        
    end
    
    % Print stats    
    if total_trials >= training_block_cumsum(sampleID) + test_block_cumsum(sampleID+1)%mod(total_trials,CONFIG.block_num)==0
%         if ~CONFIG.pre
%             CONFIG.n_distractors = CONFIG.n_distractors+1;
%         end
%         updatePattern = true;
%         if CONFIG.n_distractors==5
%             waterTime = waterTime+0.02;
%         end
%         if CONFIG.n_distractors>5
%             CONFIG.n_distractors = 1;
%         end
        
        sampleID = sampleID + 1;
%         testBlock = false;       
        if mod(sampleID,2)==0
            targetID = sampleID - 1;
        else
            targetID = sampleID + 1;
        end
    elseif total_trials >= training_block_cumsum(sampleID) + test_block_cumsum(sampleID) && ...
            total_trials < training_block_cumsum(sampleID) + test_block_cumsum(sampleID+1)
%         if total_trials<CONFIG.n_pairs*CONFIG.block_num
%             if targetID>CONFIG.n_pairs/2
%                 targetID = 1;
%             end
%         else
%             if targetID>CONFIG.n_pairs
%                 targetID = CONFIG.n_pairs/2+1;
%             end
%         end
%         updatePattern = true;
%         testBlock = true;
        ind = total_trials - training_block_cumsum(sampleID) - test_block_cumsum(sampleID) + 1;
        testSeq = testSeqs{sampleID/2};
        targetID = testSeq(ind);
    end
    
%     if total_trials >= CONFIG.total_trials
%         expEnd = now;
%     end
    
    % Trial error
    if TRIAL.error_type %&& CONFIG.verbose
        fprintf('%s\n',TRIAL.error_msg)
        fprintf('Failed Trial.\nTrials: %d/%d\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / total_trials);
    end
    
    % End session
    if escape_pressed
        break;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                         Session End                         %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if CONFIG.verbose
    fprintf('-----------------------------------------------------\n');
    fprintf('Session ended\n');
    fprintf('Successful trials: %d/%d\n\nSuccess rate: %.2f%%\n', successful_trials, total_trials, 100 * successful_trials / total_trials);
end
SESSION.timer_session_end = GetSecs - timer_session_start;
TouchQueueStop(dev);
TouchQueueRelease(dev);
RestrictKeysForKbCheck([]);
ShowCursor(w);
sca;

% Save results
if CONFIG.save_results
    % Save the last TRIALS file
    trials = TRIALS(1:total_trials);
    save(filepath,'trials','timer_session_start','iShuffle');
%     if mod(total_trials, CONFIG.n_trials_to_save)
%         TRIALS_filenames = vertcat(TRIALS_filenames, {sprintf('%s/%s_trials_%.4d_to_%.4d.mat', ...
%             CONFIG.results_dir, filename, 1 + (size(TRIALS_filenames, 1) * CONFIG.n_trials_to_save), total_trials)});
%         % Delete the trials in which trial_number == 0
%         TRIALS(~vertcat(TRIALS(1:end).trial_number)) = [];
%         % Save the last trial
%         save(TRIALS_filenames{end,:}, 'TRIALS')
%     end
%     
%     % Concatenate TRIALS files and save it under SESSION
%     for i_files = 1 : size(TRIALS_filenames, 1)
%         temp_trials = load(TRIALS_filenames{i_files, :});
%         SESSION.TRIALS = [SESSION.TRIALS(1:end) temp_trials(1:end).TRIALS];
%     end
%     % Save TRIALS under SESSION
%     save(sprintf('%s/%s.mat',CONFIG.results_dir, filename),'SESSION', 'CONFIG');
%     % Delete the temporary TRIALS filenames
%     for i_files = 1 : size(TRIALS_filenames, 1)
%         delete(TRIALS_filenames{i_files});
%     end
end

% Close audio devices
if CONFIG.play_sound
    stop_all_sound_effects(PAHANDLES, CONFIG);
    terminate_all_sound_effects(PAHANDLES, CONFIG);
end

end

%%
function [coords_all_objs, library_objs] = calculate_coords_polygons (CONFIG)

library_objs = {'circle', 'rect', 'square', 'triangle', 'cross', 'octagon',...
    'virus', 'spaceship', 'pentagon'};

% Floor radius_square
radius_square = floor(CONFIG.half_side_square);

% Calculate centers of object positions (matlab linear indexing)
center_screen = [CONFIG.main_display_area(3)-CONFIG.main_display_area(1) ...
    CONFIG.main_display_area(4)-CONFIG.main_display_area(2)] / 2;
switch CONFIG.n_distractors
    case 0 % 4 possible positions
        angles = [45 135 225 315];
        phase = angles(randperm(4,1)); % Determine the angles of the positions randomly
        
    case 1
        phase = 0;
        
    case 2
        phase = -30;
        
    case 3
        phase = -45;
        
    case 4
        phase = 18;
        
    case 5
        phase = 0;
        
    otherwise
        error("Invalid number of distractors");
end

centers_objs = zeros(CONFIG.n_distractors + 2, 2);
for i = 1 : CONFIG.n_distractors + 1
    % The positions will lay on a circle and x and y will the the cos and sines of the angles formed by the origin, x and the radius of the circle.
    centers_objs(i,:) = center_screen + [cos(deg2rad(360 / (CONFIG.n_distractors + 1) * i + phase)), -sin(deg2rad(360 / (CONFIG.n_distractors + 1) * i + phase))] * CONFIG.distance_from_center_x_y;
end
centers_objs(i + 1,:) = center_screen; % The position of the object in the center of the screen

% Onset polygon / circle
n_sides = 360;
angles_deg = linspace(0, 360, n_sides); % Angles at which our polygon vertices endpoints will be. It's inscribed in a circle
angles_rad = angles_deg * (pi / 180);
% X and Y coordinates of the points defining out polygon, centred on the
% centre of the screen
xPosVector = cos(angles_rad) .* CONFIG.radius_onset_polygon + center_screen(1);
yPosVector = sin(angles_rad) .* CONFIG.radius_onset_polygon + center_screen(2);
coords_onset_pol(1, 1 : 2 : n_sides * 2) = xPosVector(1:end);
coords_onset_pol(1, 2 : 2 : n_sides * 2) = yPosVector(1:end);

% Rects touch
coords_rect_touch = zeros(CONFIG.n_distractors + 2, 4);
for i = 1 : CONFIG.n_distractors + 2
    coords_rect_touch(i, 1:2) = centers_objs(i,:) + [-1 -1] * CONFIG.rect_touch_half_side; % x y top left vertice
    coords_rect_touch(i, 3:4) = centers_objs(i,:) + [ 1  1] * CONFIG.rect_touch_half_side; % x y bottom right vertice
end

% Squares
coords_squares = zeros(CONFIG.n_distractors + 2,8);
for i = 1 : CONFIG.n_distractors + 2
    coords_squares(i, 1:2) = centers_objs(i,:) + [-1 -1] * radius_square; % x y top left vertice
    coords_squares(i, 3:4) = centers_objs(i,:) + [-1  1] * radius_square; % x y bottom left vertice
    coords_squares(i, 5:6) = centers_objs(i,:) + [ 1  1] * radius_square; % x y bottom right vertice
    coords_squares(i, 7:8) = centers_objs(i,:) + [ 1 -1] * radius_square; % x y top right vertice
%     coords_squares(i,:) = [1920 1080 1920 1080 1920 1080 1920 1080] - coords_squares(i,:);
end

% Triangles
coords_triangles = zeros(CONFIG.n_distractors + 2,6);
for i = 1 : CONFIG.n_distractors + 2
    coords_triangles(i, 1:2) = centers_objs(i, :) + [ 0 -1] * radius_square; % top
    coords_triangles(i, 3:4) = centers_objs(i, :) + [-1  1] * radius_square; % x y bottom left vertice
    coords_triangles(i, 5:6) = centers_objs(i, :) + [ 1  1] * radius_square; % x y bottom right vertice
end

% Crosses
coords_crosses = zeros(CONFIG.n_distractors + 2,24);
for i = 1 : CONFIG.n_distractors + 2
    coords_crosses(i, 1:2)   = centers_objs(i, :) + [ 1/2 -1  ] * radius_square; % x y head right vertice
    coords_crosses(i, 3:4)   = centers_objs(i, :) + [-1/2 -1  ] * radius_square; % head left vertice
    coords_crosses(i, 5:6)   = centers_objs(i, :) + [-1/2 -1/2] * radius_square; % head left-arm vertice
    coords_crosses(i, 7:8)   = centers_objs(i, :) + [-1   -1/2] * radius_square; % left-arm top vertice
    coords_crosses(i, 9:10)  = centers_objs(i, :) + [-1    1/2] * radius_square; % left-arm bottom vertice
    coords_crosses(i, 11:12) = centers_objs(i, :) + [-1/2  1/2] * radius_square; % left-arm tail vertice
    coords_crosses(i, 13:14) = centers_objs(i, :) + [-1/2  1  ] * radius_square; % tail left vertice
    coords_crosses(i, 15:16) = centers_objs(i, :) + [ 1/2  1  ] * radius_square; % tail right vertice
    coords_crosses(i, 17:18) = centers_objs(i, :) + [ 1/2  1/2] * radius_square; % tail right-arm vertice
    coords_crosses(i, 19:20) = centers_objs(i, :) + [ 1    1/2] * radius_square; % right-arm bottom vertice
    coords_crosses(i, 21:22) = centers_objs(i, :) + [ 1   -1/2] * radius_square; % right-arm top vertice
    coords_crosses(i, 23:24) = centers_objs(i, :) + [ 1/2 -1/2] * radius_square; % right-arm head vertice
end

% Octagons
coords_octagons = zeros(CONFIG.n_distractors + 2,16);
for i = 1 : CONFIG.n_distractors + 2
    coords_octagons(i, 1:2)   = centers_objs(i, :) + [ 1/4 -1  ] * radius_square; % x y head right vertice
    coords_octagons(i, 3:4)   = centers_objs(i, :) + [-1/4 -1  ] * radius_square; % head left vertice
    coords_octagons(i, 5:6)   = centers_objs(i, :) + [-1   -1/4] * radius_square; % left-arm top vertice
    coords_octagons(i, 7:8)   = centers_objs(i, :) + [-1    1/4] * radius_square; % left-arm bottom vertice
    coords_octagons(i, 9:10)  = centers_objs(i, :) + [-1/4  1  ] * radius_square; % tail left vertice
    coords_octagons(i, 11:12) = centers_objs(i, :) + [ 1/4  1  ] * radius_square; % tail right vertice
    coords_octagons(i, 13:14) = centers_objs(i, :) + [ 1    1/4] * radius_square; % right-arm bottom vertice
    coords_octagons(i, 15:16) = centers_objs(i, :) + [ 1   -1/4] * radius_square; % right-arm top vertice
end

% Viruses
coords_viruses = zeros(CONFIG.n_distractors + 2,40);
for i = 1 : CONFIG.n_distractors + 2
    coords_viruses(i, 1:2)   = centers_objs(i, :) + [-1/2 -1/2] * radius_square;
    coords_viruses(i, 3:4)   = centers_objs(i, :) + [-1/2 -1  ] * radius_square;
    coords_viruses(i, 5:6)   = centers_objs(i, :) + [-1    -1 ] * radius_square;
    coords_viruses(i, 7:8)   = centers_objs(i, :) + [-1   -1/2] * radius_square;
    coords_viruses(i, 9:10)  = centers_objs(i, :) + [-1/2 -1/2] * radius_square;
    coords_viruses(i, 11:12) = centers_objs(i, :) + [-1/2  1/2] * radius_square;
    coords_viruses(i, 13:14) = centers_objs(i, :) + [-1    1/2] * radius_square;
    coords_viruses(i, 15:16) = centers_objs(i, :) + [-1    1  ] * radius_square;
    coords_viruses(i, 17:18) = centers_objs(i, :) + [-1/2  1  ] * radius_square;
    coords_viruses(i, 19:20) = centers_objs(i, :) + [-1/2  1/2] * radius_square;
    coords_viruses(i, 21:22) = centers_objs(i, :) + [ 1/2  1/2] * radius_square;
    coords_viruses(i, 23:24) = centers_objs(i, :) + [ 1/2  1  ] * radius_square;
    coords_viruses(i, 25:26) = centers_objs(i, :) + [ 1    1  ] * radius_square;
    coords_viruses(i, 27:28) = centers_objs(i, :) + [ 1    1/2] * radius_square;
    coords_viruses(i, 29:30) = centers_objs(i, :) + [ 1/2  1/2] * radius_square;
    coords_viruses(i, 31:32) = centers_objs(i, :) + [ 1/2 -1/2] * radius_square;
    coords_viruses(i, 33:34) = centers_objs(i, :) + [ 1   -1/2] * radius_square;
    coords_viruses(i, 35:36) = centers_objs(i, :) + [ 1   -1  ] * radius_square;
    coords_viruses(i, 37:38) = centers_objs(i, :) + [ 1/2 -1  ] * radius_square;
    coords_viruses(i, 39:40) = centers_objs(i, :) + [ 1/2 -1/2] * radius_square;
end

% Spaceships
coords_spaceships = zeros(CONFIG.n_distractors + 2,24);
for i = 1 : CONFIG.n_distractors + 2
    coords_spaceships(i, 1:2)   = centers_objs(i, :) + [-1/2 -1  ] * radius_square;
    coords_spaceships(i, 3:4)   = centers_objs(i, :) + [-1/2 -1/2] * radius_square;
    coords_spaceships(i, 5:6)   = centers_objs(i, :) + [-1   -1/2] * radius_square;
    coords_spaceships(i, 7:8)   = centers_objs(i, :) + [-1    1  ] * radius_square;
    coords_spaceships(i, 9:10)  = centers_objs(i, :) + [-1/2  1  ] * radius_square;
    coords_spaceships(i, 11:12) = centers_objs(i, :) + [-1/2  1/2] * radius_square;
    coords_spaceships(i, 13:14) = centers_objs(i, :) + [ 1/2  1/2] * radius_square;
    coords_spaceships(i, 15:16) = centers_objs(i, :) + [ 1/2  1  ] * radius_square;
    coords_spaceships(i, 17:18) = centers_objs(i, :) + [ 1    1  ] * radius_square;
    coords_spaceships(i, 19:20) = centers_objs(i, :) + [ 1   -1/2] * radius_square;
    coords_spaceships(i, 21:22) = centers_objs(i, :) + [ 1/2 -1/2] * radius_square;
    coords_spaceships(i, 23:24) = centers_objs(i, :) + [ 1/2 -1  ] * radius_square;
end

% Pentagon
coords_pentagons = zeros(CONFIG.n_distractors + 2,10);
for i = 1 : CONFIG.n_distractors + 2
    coords_pentagons(i, 1:2) = centers_objs(i,:) + [0 -1] * radius_square; % x y top left vertice
    coords_pentagons(i, 3:4) = centers_objs(i,:) + [-1  0] * radius_square; % x y bottom left vertice
    coords_pentagons(i, 5:6) = centers_objs(i,:) + [-1  1] * radius_square; % x y bottom right vertice
    coords_pentagons(i, 7:8) = centers_objs(i,:) + [1  1] * radius_square; % x y top right vertice
    coords_pentagons(i, 9:10) = centers_objs(i,:) + [1  0] * radius_square; % x y top right vertice
end

% Create a cell with reshaped to Screen('fillRect') or Screen('fillPoly')
coords_all_objs = cell(1, 9);
coords_all_objs{1} = reshape(coords_onset_pol,[2, size(coords_onset_pol,2) / 2])'; % Onset polygon / circle
coords_all_objs{2} = coords_rect_touch(1:end,1:4)'; % Rect touches
for i = 1 : CONFIG.n_distractors + 2
    coords_all_objs{3}{i,1} = reshape(coords_squares(i,:),[2,4])'; % Squares
    coords_all_objs{4}{i,1} = reshape(coords_triangles(i,:),[2,3])'; % Triangles
    coords_all_objs{5}{i,1} = reshape(coords_crosses(i,:),[2,12])'; % Crosses
    coords_all_objs{6}{i,1} = reshape(coords_octagons(i,:),[2,8])'; % Octagons
    coords_all_objs{7}{i,1} = reshape(coords_viruses(i,:),[2,20])'; % Viruses
    coords_all_objs{8}{i,1} = reshape(coords_spaceships(i,:),[2,12])'; % Spaceships
    coords_all_objs{9}{i,1} = reshape(coords_pentagons(i,:),[2,5])'; % Pentagons
end

end

%%
function update_touch_log (CONFIG, filename, TRIAL)

% Create a .txt file to log touch behavior in real time.
% Evt.type: -1 = touch outside the objects.
%            0 = no touch.

% To create results dir
if ~exist(CONFIG.results_dir, 'dir')
    mkdir(CONFIG.results_dir)
end

% To create the header of the file
if ~exist(sprintf('%s/%s_training.txt', CONFIG.results_dir, filename),'file')
    fileID = fopen(sprintf('%s/%s_training.txt',CONFIG.results_dir, filename),'a+');
    formatSpec = '%6s%8s%6s%13s%24s%14s%13s%12s%11s%9s%10s%14s%9s\n\n';
    text = {'trial', 'period', 'x', 'y', 'intertrial_start', 'sample_start', 'delay_start',...
        'test_start', 'rt_test', 'rt_test', 'evt_type', 'touch_obj_id', 'success'};
    fprintf(fileID,formatSpec, text{:});
    fclose(fileID);
else
    % To add data
    fileID = fopen(sprintf('%s/%s_training.txt',CONFIG.results_dir, filename),'a+');
    formatSpec = '%4d,%6d,%12.4f,%12.4f,%12d,%19.4f,%12.4f,%11.4f,%10.4f,%8.4f,%7d,%11d,%10d\n';
    for i_rows = 1:size(TRIAL.touch_log,1)
        fprintf(fileID,formatSpec, TRIAL.touch_log(i_rows,:));
    end
    fclose(fileID);
    
end
end

%%
function give_juice(a, duration)
% To give juice for a specific duration

% Write 1 to digital pin 'which_pin' to turn the water pump on.
writeDigitalPin(a, 'D2', 1);
% Duration of the pulse in seconds
WaitSecs(duration);
% Write 0 to digital pin 'which_pin' to turn the water pump on.
writeDigitalPin(a, 'D2', 0);

end

%%
function flicker_screen_punishment(w, CONFIG)
black = BlackIndex(CONFIG.main_display);
white = WhiteIndex(CONFIG.main_display);
flickerBegin = GetSecs;
while GetSecs-flickerBegin<CONFIG.FlickerPunishment
% for i_flicker = 1 : 3
    Screen('FillRect', w, white); % To show a white screen
    Screen('Flip', w);
    Screen('Flip', w);
    Screen('FillRect', w, black);
    Screen('Flip', w);
    Screen('Flip', w);
end

end

%%
function [PAHANDLES] = setup_sound_effects(PAHANDLES, CONFIG)

% To stop execution of the function if the flag doPlaySound is called
if ~CONFIG.play_sound
    return;
end

[soundEffect_start_trial, freq] = audioread(fullfile(CONFIG.main_dir, 'Beep_clickCorrected.wav'));
[soundEffect_fail] = audioread(fullfile(CONFIG.main_dir, 'Bump_clickCorrected.wav'));
[soundEffect_success] = audioread(fullfile(CONFIG.main_dir, 'Coin_clickCorrected.wav'));

InitializePsychSound;
% Force GetSecs and WaitSecs into memory to avoid latency later on:
GetSecs;
WaitSecs(0.1);

deviceid = 8;%[];% Details can be found in PsychPortAudioTimingTest.m
reqlatencyclass = 2;
suggestedLatencySecs = 0.015;
nrchannels = 1;
% repetitions = 1;
buffersize = 0.05; % It influences the latency of the sound. A bigger number will delay the delivery of the sound
% pahandle = PsychPortAudio('Open', deviceid, [], reqlatencyclass, freq, nrchannels, buffersize, suggestedLatencySecs);
PAHANDLES.start_trial = PsychPortAudio('Open', deviceid, [], 0, freq, nrchannels, reqlatencyclass, buffersize, suggestedLatencySecs);
PAHANDLES.fail = PsychPortAudio('Open', deviceid, [], 0, freq, nrchannels, reqlatencyclass, buffersize, suggestedLatencySecs);
PAHANDLES.success = PsychPortAudio('Open', deviceid, [], 0, freq, nrchannels, reqlatencyclass, buffersize, suggestedLatencySecs);

% Fill the audio playback buffer with the audio data 'wavedata':
% [underflow, nextSampleStartIndex, nextSampleETASecs] = PsychPortAudio(�FillBuffer�?? pahandle, bufferdata [, streamingrefill=0][, startIndex=Append]);
PsychPortAudio('FillBuffer', PAHANDLES.start_trial, soundEffect_start_trial');
PsychPortAudio('FillBuffer', PAHANDLES.fail, soundEffect_fail');
PsychPortAudio('FillBuffer', PAHANDLES.success, soundEffect_success');

% Perform one warmup trial, to get the sound hardware fully up and running,
% performing whatever lazy initialization only happens at real first use.
% This "useless" warmup will allow for lower latency for start of playback
% during actual use of the audio driver in the real trials:
PsychPortAudio('Start', PAHANDLES.start_trial, 1, 0, 1);
WaitSecs(0.5);
PsychPortAudio('Start', PAHANDLES.fail, 1, 0, 1);
WaitSecs(0.5);
PsychPortAudio('Start', PAHANDLES.success, 1, 0, 1);

end

function play_sound_effect(pahandle, CONFIG)
% To stop execution of the function if the flag doPlaySound is called
if ~~CONFIG.play_sound
    return;
end

PsychPortAudio('Start', pahandle, 1, 0, 1);

end


%%
function stop_all_sound_effects(PAHANDLES, CONFIG)

% To stop execution of the function if the flag doPlaySound is called
if ~CONFIG.play_sound
    return;
end

PAHANDLES_fieldnames = fieldnames(PAHANDLES);
if ~isempty(PAHANDLES_fieldnames)
    for i = 1 : length(PAHANDLES_fieldnames)
        pahandle = eval(strcat('PAHANDLES.', PAHANDLES_fieldnames{i}));
        if ~isempty(pahandle)
            PsychPortAudio('STOP',pahandle, 1); % To wait for any sound to finish playing
        end
    end
end

end

%%
function terminate_all_sound_effects(PAHANDLES, CONFIG)

% To stop execution of the function if the flag doPlaySound is called
if ~CONFIG.play_sound
    return;
end

PAHANDLES_fieldnames = fieldnames(PAHANDLES);
if ~isempty(PAHANDLES_fieldnames)
    for i = 1 : length(PAHANDLES_fieldnames)
        pahandle = eval(strcat('PAHANDLES.', PAHANDLES_fieldnames{i}));
        if ~isempty(pahandle)
            PsychPortAudio('Close', eval(strcat('PAHANDLES.', PAHANDLES_fieldnames{i})));
        end
    end
end

end

%%
function save_copy_code(folderpath,fname)
codepath = mfilename('fullpath');
codepath = [codepath '.m'];
[~,codename,~] = fileparts(codepath);
filename = [fname '-' codename '.m'];
copypath = fullfile(folderpath,filename);
copyfile(codepath,copypath);

end