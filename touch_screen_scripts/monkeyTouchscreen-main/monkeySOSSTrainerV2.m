function monkeySOSSTrainerV2(setupID, monkeyID)
% Cognitive task follow the training protocol
% _________________________________________________________________________
%
% History:
% 3-Nov-2020 coded based on touchMultiRect and MAIN_touch_square.
% 23-Jul-2022 modified from monkeySOSSTrainer
 
% _________________________________________________________________________
if ~exist('setupID','var') || isempty(setupID)
    setupID = 'xxx';
end
if ~exist('monkeyID','var') || isempty(monkeyID)
    monkeyID = 'xxxx';
end

% Autodetect an Arduino Uno.
% pkg load arduino
try
    ard = arduino();
    give_juice(ard, 0.1);
catch
    ard = [];
end


% Setup useful PTB defaults:
PsychDefaultSetup(2);

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
% setupType=0;
CONFIG.scale_touchscreen=1/2;
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
    case 4 % tengyu with NUC10
        CONFIG.main_display_area = [1920 0 3840 1080];
        touchscreenID = 'ILITEK ILITEK-TP';
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
water_time = 0.06;
data_dir = fullfile(pwd,'Results');
CONFIG.main_display = max(Screen('Screens'));% Number of the surface screen
CONFIG.blockSwitcher = 0; % 0 block switch by total trials in current block
% 1 block switch by correct trials in current block
CONFIG.flicker = true;
CONFIG.beepon = true;
CONFIG.testOrder = false;
CONFIG.trialInterval = 0.4;%1/2;
CONFIG.touchInterval = 8; % When multiple squares are displayed, if the monkey has touched one, he should touched another one in 'touchInterval' seconds.
% time_inbetween_touches = 0.01;
CONFIG.show_map_squares = 0;
% CONFIG.pretrainingTrials = inf;%10;%0;%inf;
CONFIG.totalTrialsSham = 6;
CONFIG.totalTrialsSOSS = 64;
CONFIG.correct_trials_per_block_sham = 6;
CONFIG.correct_trials_per_block_soss = 64;
CONFIG.pre = 0; % 0 test; 1 first pre-stage; 2 second pre-stage

% Find the color values which correspond to white and black: Usually
% black is always 0 and white 255, but this rule is not true if one of
% the high precision framebuffer modes is enabled via the
% PsychImaging() commmand, so we query the true values via the
% functions WhiteIndex and BlackIndex:
% white=WhiteIndex(screenNumber);
% black=BlackIndex(screenNumber);
black = [0 0 0];
% gray=(white + black) / 2;
gray = [0.5 0.5 0.5];
red = [1.0 0 0];
cyan = [0 1.0 1.0];
green = [0 1.0 0];
orange = [255 127 0]/255;

switch CONFIG.pre
    case 0
        pretrainingTrials = 0;
        maximumSessionNum = 8;
    case 1
        pretrainingTrials = inf;
        maximumSessionNum = 1;
    case 2
        pretrainingTrials = 10;
        maximumSessionNum = 11;%inf;
end
% pretrainingTrials = CONFIG.pretrainingTrials;
totalTrialsSham = CONFIG.totalTrialsSham;
totalTrialsSOSS = CONFIG.totalTrialsSOSS;
correct_trials_per_block_sham = CONFIG.correct_trials_per_block_sham;
correct_trials_per_block_soss = CONFIG.correct_trials_per_block_soss;
blockSwitcher = CONFIG.blockSwitcher;
flicker = CONFIG.flicker;
beepon = CONFIG.beepon;
testOrder = CONFIG.testOrder;
trialInterval = CONFIG.trialInterval;
touchInterval = CONFIG.touchInterval;

dev = GetTouchDeviceIndices([], 1, touchscreenID);

% Open a default onscreen window with black background color and 0-1 color
% range:
[w, rect] = PsychImaging('OpenWindow', CONFIG.main_display, black, CONFIG.main_display_area);
blackBG = true;
% textureIndex = Screen('MakeTexture', w, fig);
            
% Get the center coordinate of the window in pixels
[xCenter, yCenter] = RectCenter(rect);

% Get the size of the on screen window in pixels
% For help see: Screen WindowSize?
[screenWidth, screenHeight] = Screen('WindowSize', w);

% Define correct audio
Fs = 44100;
pahandle = PsychPortAudio('Open', 8, [], 2, Fs);
beepWaveform = MakeBeep(500, 0.3, Fs);
beepWaveform = repmat(beepWaveform,2,1);
PsychPortAudio('FillBuffer', pahandle, beepWaveform);

% The side length of a square is 20 mm
rectWidth = round(20/pixel2mm); 
rectHeight = round(20/pixel2mm);
gapLen = round(3/pixel2mm); % The gap between two adjacent rectangles is 3 mm

[visible_squares, visible_squares_coords] = calculate_squares(screenWidth,...
    screenHeight,6,4,rectWidth,rectHeight,gapLen,0,0);    


% % pkg load statistics
% rand("seed",100*sum(clock))
rng(sum(100*clock));
shiftRects = visible_squares_coords(:,1:4);


% Set the color of our square to full red. Color is defined by red green
% and blue components (RGB). So we have three numbers which
% define our RGB values. The maximum number for each is 1 and the minimum
% 0. So, "full red" is [1 0 0]. "Full green" [0 1 0] and "full blue" [0 0
% 1]. Play around with these numbers and see the result.
fps = Screen('NominalFrameRate',w);
% colors = [linspace(rectColor(1),1,round(fps))' linspace(rectColor(2),0,round(fps))'...
%     linspace(rectColor(3),0,round(fps))'];

% Get maximum supported dot diameter for smooth dots:a
[~, maxSmoothPointSize] = Screen('DrawDots', w);

% Select good diameter for touch point blobs, but no more than what
% 'DrawDots' supports:
baseSize = min(RectWidth(rect) / 40, maxSmoothPointSize);

% [Optional] To show a map of the squares and square IDs before the program starts
if CONFIG.show_map_squares
    Screen('FillRect', w, gray, visible_squares_coords(:,1:4)');
    for i_obj = 1 : visible_squares_coords(end, 6)
        Screen('DrawText', w, sprintf('%d', i_obj), visible_squares_coords(i_obj, 1) + rectWidth/2, visible_squares_coords(i_obj, 2) + rectHeight/2, red);
    end
    Screen('Flip', w);
    WaitSecs(5);
end
   
%% Function to process touch events
    function process_touch_event()
        % TouchEventAvail reports the number of events in a touch queue.
        % One single touch can contain many events.
        while TouchEventAvail(dev)
            
            evt_count = evt_count + 1;
            
            %  Return oldest pending event
            evt = TouchEventGet(dev, w);
            
            evtNum = evtNum + 1;
            tdata(evtNum,:) = [evt.MappedX, evt.MappedY evt.Time evt.Type];
            
            % Touch blob id - Unique in the session at least as
            % long as the finger stays on the screen:
            id = evt.Keycode;
            
            % Only consider the id of the first touch event
            if evt_count == 1
                first_event_id = id;
            end
            
            %             if id == first_event_id
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
%                     if patternDisplayed
%                         Screen('FillRect', w, rectColors', displayedRects');
%                     end
                    Screen('DrawDots', w, [evt.MappedX; evt.MappedY], ...
                        baseSize, [1,1,1], [], 1, 1);
%                     Screen('Flip', w);
                    %                         patternDisplayed = 0;
                    
                case {2, 3}
                    % 2: New touch point -> New blob!
                    % 3: Moving touch point -> Moving blob!
                    if id == first_event_id
                        blob.mul = 1.0; % size of the blob
                        blob.x = evt.MappedX;
                        blob.y = evt.MappedY;
                        blob.t = evt.Time;
                    end
                    monkey_touching_screen = 1;
                    
                case 4
                    % Touch released -> Dying blob!
                    if id == first_event_id
                        blob.mul = 0;
                        blob.x = evt.MappedX;
                        blob.y = evt.MappedY;
                    end
                    monkey_touching_screen = 2;
                    
                case 5
                    % Lost touch data for some reason:
                    % Flush screen red for one video refresh cycle.
                    fprintf(['Ooops - Sequence data loss! 3rd party ' ...
                        'interference or overload?\n']);
                    Screen('FillRect', w, [1 0 0]);
                    Screen('Flip', w);
                    Screen('FillRect', w, 0);
%                     Screen('FillRect', w, rectColors', displayedRects');
%                     Screen('Flip', w);
                    
            end
            %             end
        end
        
        % Now that all touches for this iteration are processed, repaint
        % the live blob in its new position or fade out a dying blob
        if ~isempty(blob) && blob.mul > 0.1
            % Draw the blob: .mul defines size of the blob:
%             Screen('DrawDots', w, [blob.x, blob.y], ...
%                 blob.mul * baseSize, orange, [], 1, 1);
        else
            % Below threshold: Kill the blob
            blob = [];
        end
        
        % To determine if blob is empty
        if evt_count && isempty(blob)
            evt_count = 0;
        end
        
        if buttonstate
            Screen('FrameRect', w, [1, 1, 0], [], 5);
        end
%         if ~isempty(blob)
%             [blob.x,blob.y]
%         end
    end


%% Function to save task data
    function saveData()
        timeInfo(3,:) = clock;
        expData = expdata(1:nSession);
        allRects = visible_squares_coords(:,1:4);
        save(filepath ,'expData','timeInfo','allRects','monkeyID')
        for jj=1:evtNum
            fprintf(fid,'%f %f %f %d \n',tdata(jj,1),tdata(jj,2),tdata(jj,3),tdata(jj,4));
        end
%         fclose(fid);
        evtNum = 0;
%         pause(10*rand(1))
%         subfolder = datestr(clock,30);
%         subfolder = subfolder(1:8);
%         mftp = ftp('10.10.1.155','monkey','Marmoset03');
%         cd(mftp,fullfile('touchData',setupID))
%         try
%             cd(mftp,subfolder)
%         catch
%             mftp.mkdir(subfolder)
%             cd(mftp,subfolder)
%         end
%         mput(mftp,filepath);
%         mput(mftp,copypath);      
%         close(mftp)
    end

%% Function to initialize parameters
    function initializePara()
        tclock = clock;
        today = tclock(1:3);
        expBegin = datenum([today 11 0 0]);
        expEnd = datenum([today 14 0 0]);
        nextDay = datenum([today 0 0 0]) + 1; 
        checkPoint = datenum([today 10 0 0]);
        
        subfolder = datestr(tclock,'yyyymmdd');
        folderpath = fullfile(data_dir,subfolder);
        if ~exist(folderpath,'dir')
            mkdir(folderpath)
        end
        fname = [setupID '-' datestr(tclock,30) '-' monkeyID];
        filename = [fname '.mat'];
        filepath = fullfile(folderpath,filename);
        filename = [fname '.txt'];
        txtpath = fullfile(folderpath,filename);
        fid = fopen(txtpath,'w');

        tdata = nan(10000,4);
        if ~isempty(expdata)
            expdata = [];
        end
        expdata.touch = nan(42,9);
        expdata.id = nan(18,4);
        expdata = repmat(expdata,200,1);
        firstTouch = true;
%         timeInfo(1,:) = tclock;
        
        nTrial = 0;
        nCorrectTrial = 0;
        nSession = 1;
        if pretrainingTrials==0
            nStep = 2;               
            touchedColor = gray;
        else
            nStep = 1;
            touchedColor = black;
        end
        nTouch = 0;
        evtNum = 0;
        
        evt_count = 0;
        first_event_id = [];
        buttonstate = 0;
        blob = [];
        monkey_touching_screen = 0;
        
        sham = 0;
        rectColor = gray;
        nRect = 2;
        [displayedRects,irect,rectColors] = setPattern(shiftRects,rectColor,nRect);
        touchOrder = [];
        ind = 1;
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                              Task                               %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

HideCursor(w);
clc
try
    % Create and start touch queue for window and device:
    TouchQueueCreate(w, dev);
    TouchQueueStart(dev);
    
    % Wait for the go!
    KbReleaseWait;
    
    % Only ESCape allows to exit the demo:
    RestrictKeysForKbCheck(KbName('ESCAPE'));
    
    % Initializations      

    folderpath = [];
    fname = [];
    filepath = [];
    fid = [];
    expBegin = [];
    expEnd = [];
    nextDay = [];
    checkPoint = [];
    timeInfo = zeros(3,6);
    tdata = [];
    expdata = [];
    touchSession = nan(42,9);      
    rectID = nan(10,5);
    evt = [];
    evt_count = 0;
    evtNum = 0;
    first_event_id = [];
    buttonstate = 0;
    blob = [];
    monkey_touching_screen = 0;
    trialBegin = 0;
    trialEnd = GetSecs;
    updatePattern = 0;
    iTouched = [];
    sham = 0;
    rectColor = [];
    patternDisplayed = 0;
    refreshTouchQueue = 0;
    flushTouch = false;
    
    initializePara();
    
    % To save a copy of the code
    copypath = copy_code(folderpath,fname);
    
%     blob = [];
%     single_evt_id = [];
%     blob_inside_rect = 0;
%     inside_bonus = 0;

    
    escKey = KbName('ESCAPE');

    
    t0 = tic;
    % Main loop: Run until keypress:
    while 1
        [~, ~, keyCode] = KbCheck;
        if keyCode(escKey)
            break;
        end
        tnow = now;
        if tnow>nextDay          
%             saveData;
            initializePara; 
            copypath = copy_code(folderpath,fname);
        end
        
        if tnow>expEnd
%             break;
            saveData;
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
%             Screen('FillRect', w, gray);
%             Screen('FillRect', w, rectColor, displayedRects');
            %           Screen('DrawText', w, sprintf('The %d-th trial (%d/%d)',trialNum,repNum,nRep), 80, 45, [255 0 0]);
            %             Screen('DrawTexture', w, textureIndex);
            % Draw the rectangle
            Screen('Flip', w);
            timeInfo(1,:) = clock;
            blackBG = false;
            TouchEventFlush(dev);
        end 
        
%         if ~flicker
%             Screen('FillRect', w, rectColor, displayedRects');
%             Screen('Flip', w);
%         else
%             jj = mod(jj-1,round(fps))+1;
%             Screen('FillRect', w, colors(jj,:), displayedRects');
%             jj = jj + 1;
%             Screen('Flip', w);
%         end
        if toc(t0)>60 % Check whether the program is stuck
            fprintf('%s\n',datestr(now))
            t0 = tic;
        end
        
%         % Flip screen
%         timer = Screen('Flip', w);
%         
%         % Timer for intertrial interval
%         timer_end = timer + CONFIG.intertrial_interval;
%         
%         % Record start of sample period (relative to session start)
%         TRIAL.intertrial_period_start = timer - timer_session_start;
%         
%         % Record start of sample period (absolute)
%         intertrial_period_start_abs = timer;
%         
%         % Timer between the recordings of 2 touches
%         timer_inbetween_touches = GetSecs - CONFIG.time_inbetween_touches;
%     
%     
%         timer = Screen('Flip', w);

        % Process all currently pending touch events:
        process_touch_event;
        
        if firstTouch&&~isempty(blob)
            timeInfo(2,:) = clock;
            firstTouch = false;
        end
        
        % Don't start trial if the monkey keeps touching the screen
        while (trialEnd&&GetSecs-trialEnd<=trialInterval)||(trialBegin&&GetSecs-trialBegin<=trialInterval)
%             Screen('Flip', w); % Necessary to clean the blob while keeping touch
            while monkey_touching_screen %&& ~escape_pressed
%                 escape_pressed = KbCheck;
                TouchEventFlush(dev);
                blob = [];
                monkey_touching_screen = 0;
                WaitSecs(0.4);
                process_touch_event;%
                if KbCheck
                    break;
                end
            end
            % Without creating a black window in the end, the path of the touching finger will be shown.
            % Ideally I should clear the framebuffer instead of using this workaround.
            %             Screen('FillRect', w, gray, rect);
%             blob = [];
%             evt = [];
%             first_event_id = [];
%             flushTouch = true;
        end
%         if flushTouch
%             TouchEventFlush(dev);
%             flushTouch = false;
%         end
        
        if ~patternDisplayed
            % Draw the rectangle
            Screen('FillRect', w, rectColors', displayedRects');
            
            % Done repainting - Show it:
            Screen('Flip', w);
            patternDisplayed = 1;
        end
        
        while isempty(blob)&&trialBegin&&GetSecs-trialBegin<=touchInterval%&&now<expEnd
            process_touch_event;
            if KbCheck
                break;
            end
        end
        
        % Record touches. Touches must have 'CONFIG.time_inbetween_touches' s interval between them
        if ~isempty(blob) && (evt.Type == 2 || evt.Type == 3)% && GetSecs >= sample_period_start_abs + time_inbetween_touches
            % Detects whether the first touch point is inside the rectangle
            iTouch = 0;
            for ii=1:nRect
                if IsInRect(blob.x, blob.y, displayedRects(ii,:))
                    iTouch = ii;
                    break;
                end
            end

            if iTouch
                if all(iTouched~=iTouch)
                    trialBegin = GetSecs;
                    trialEnd = 0;
                    iTouched = [iTouched iTouch];
                    errorType = 0;
                    if ~sham
                        if isempty(touchOrder)||iTouch==touchOrder(ind)
                            rectColors(iTouch,:) = touchedColor;
                            if beepon
                                PsychPortAudio('Start', pahandle, 1);
                            end
%                             if ~isempty(ard)
%                                 give_juice(ard, waterTime/2, 1);
%                             end
                            fprintf('The %d/%d square touched!\n',length(iTouched),nRect)
                            flicker_touched_target(w,rectColors,displayedRects,iTouch);
                            %                         give_juice(ard, waterTime, 1);
                            ind = ind + 1;
                        else
                            errorType = 4;% wrong order
                            %                     nTrial = nTrial + 1;
                            updatePattern = 1;
                            %                     iTouched = [];
                            %                     trialBegin = 0;
                            %                     trialEnd = GetSecs;
                            if flicker
                                flicker_screen_punishment(w,black,red);
                            end
                        end
                    end
%                     % Self order matching
%                     if nCorrectTrial==correct_trials_per_block_sham
%                         rectColors(iTouch,:) = green;
%                         flicker_touched_target(w,rectColors,displayedRects,iTouch);
%                     end
%                     if nCorrectTrial>correct_trials_per_block_sham
%                         if iTouch==touchOrder(ind)
%                             flicker_touched_target(w,rectColors,displayedRects,iTouch);
%                             ind = ind + 1;
%                         else
%                             errorType = 4;  % wrong order
%                             updatePattern = 1;
%                             if flicker
%                                 flicker_screen_punishment(w,black,red);
%                             end
%                         end
%                     end
                    
                else
                    errorType = 1;  % touch a touched square
%                     nTrial = nTrial + 1;
                    updatePattern = 1;
%                     iTouched = [];
%                     trialBegin = 0;
%                     trialEnd = GetSecs;
                    if flicker
                        flicker_screen_punishment(w,black,red);
                    end
                end
            else
                errorType = 2; % touch outside target
                %                     nTrial = nTrial + 1;
                updatePattern = 1;
                %                     trialBegin = 0;
                %                     trialEnd = GetSecs;
                %                     iTouched = [];
                if flicker
                    flicker_screen_punishment(w,black,red);
                end               
            end
            if length(iTouched) == nRect
                if ~isempty(ard)
%                     if sham
%                         if mod(nCorrectTrial,2)==1
%                             give_juice(ard, waterTime, 1);
%                         end
%                     else
%                         give_juice(ard, waterTime, 1);
%                     end
                    give_juice(ard, water_time, 1);
                else
                    fprintf('Correct\n')
                end
%                 trialBegin = 0;
%                 trialEnd = GetSecs;
%                 nTrial = nTrial + 1;
                updatePattern = 1;
                nCorrectTrial = nCorrectTrial + 1;
                touchOrder = iTouched;
%                 if nCorrectTrial == correct_trials_per_block_sham + 1
%                     rectColors = repmat(gray,nRect,1);
%                 end
%                 iTouched = [];
            end                      
            
            t = etime(clock, timeInfo(2,:));
            nTouch = nTouch + 1;
            touchSession(nTouch,:) = [blob.x, blob.y t nTouch errorType sham nTrial nSession nStep];

            
            blob = [];
            evt = [];
%             first_event_id = [];
        else
            % Below threshold: Kill the blob:
%             TouchEventFlush(dev);

            if trialBegin&&GetSecs-trialBegin>touchInterval
                errorType = 3; %time out
                t = etime(clock, timeInfo(2,:));
                nTouch = nTouch + 1;
                touchSession(nTouch,:) = [nan, nan t nTouch errorType sham nTrial nSession nStep];
%                 trialBegin = 0;
%                 trialEnd = GetSecs;
%                 nTrial = nTrial + 1;
                updatePattern = 1;
%                 iTouched = [];
                if flicker
                    flicker_screen_punishment(w,black,red);
                end
            end
        end

            
        if updatePattern
            trialBegin = 0;
            trialEnd = GetSecs;
            iTouched = [];
            ind = 1;
            nTrial = nTrial + 1;
            rectID(nTrial,1:nRect) = irect;
            Screen('Flip', w);
            Screen('Flip', w);
            
            fprintf('The %d-th trial of %d-th session (%d correct)\n',nTrial,nSession,nCorrectTrial)
            % Training protocol
            errorTypes = touchSession(:,5);
            errorTypes = errorTypes(~isnan(errorTypes));
            if blockSwitcher && ~isnan(nCorrectTrial)
                trialNum = nCorrectTrial;
                trialNumSham = correct_trials_per_block_sham;
                trialNumSOSS = correct_trials_per_block_soss;
            else
                trialNum = nTrial;
                trialNumSham = totalTrialsSham;
                trialNumSOSS = totalTrialsSOSS;
            end
            
            if nStep==1
                touchOrder = [];
                if mod(trialNum,42)==0
                    refreshTouchQueue = 1;
                end
                if trialNum==pretrainingTrials
                    touchedColor = gray;
                    nTrial = 0;
                    nCorrectTrial = 0;
                    fprintf('Session: %d  Step: %d\n',nSession,nStep)
                    refreshTouchQueue = 1;
                    expdata(nSession).touch = touchSession(1:nTouch,:);
                    touchSession = nan(42,9);
                    expdata(nSession).id = rectID;
                    nSession = nSession + 1;
%                     touchOrder = [];
                    nTouch = 0;
                    nStep = 2;
                    pause(1)
%                     expEnd = now;
                end
            elseif nStep<=5
                if ~testOrder
                    touchOrder = [];
                end
                if sham
                    if trialNum==trialNumSham+trialNumSOSS
                        sham = false;
                        rectColor = gray;
                                            
%                         switch nStep
%                             case 1
%                                 touchedColor = black;
%                             case 2
%                                 touchedColor = gray;
%                         end

%                         shamNum = nTrial;
%                         correctShamNum = nCorrectTrial;
                        nTrial = 0;
                        nCorrectTrial = 0;
                        fprintf('Session: %d  Step: %d\n',nSession,nStep)
                        refreshTouchQueue = 1;
                        expdata(nSession).touch = touchSession(1:nTouch,:);
                        touchSession = nan(42,9);
                        expdata(nSession).id = rectID;
                        nSession = nSession + 1;
                        touchOrder = [];
                        nTouch = 0;
                        
                        if CONFIG.pre==0
                            nStep = nStep + 1;
                        end
                        if nStep>5
                            nStep = 2;
                        end
                        nRect = nStep;
                        
                        pause(1)
                        if nSession>maximumSessionNum
                            expEnd = now;
%                             nStep = 6;
%                             sham = true;
%                             rectColor = cyan;
%                             nRect = 1;
%                             rectID = nan(34,5);
                        end
                    end
                else
                    if trialNum==trialNumSOSS%+shamNum
                        sham = true;
                        rectColor = cyan;
                        nRect = 1;
%                         if (nCorrectTrial-correctShamNum)/(nTrial - shamNum)>0.7
%                             nStep = nStep + 1;
%                         elseif (nCorrectTrial-correctShamNum)/(nTrial - shamNum)<0.3
%                             nStep = nStep - 1;
%                         end
%                         if nStep>2
%                             nStep = 2;
%                         end
%                         if nStep<1
%                             nStep = 1;
%                         end
%                         nCorrectTrial
%                         if all(errorTypes==0)
%                             nStep = nStep + 1;
%                         elseif all(errorTypes>0)
%                             if nStep>1
%                                 nStep = nStep - 1;
%                             end
%                         end

%                         nTrial = 0;
%                         nCorrectTrial = 0;
%                         shamNum = 0;
%                         correctShamNum = 0;
%                         fprintf('Session: %d  Step: %d\n',nSession,nStep)
%                         refreshTouchQueue = 1;
%                         expdata(nSession).touch = touchSession(1:nTouch,:);
%                         touchSession = nan(42,9);
%                         expdata(nSession).id = rectID;
%                         nSession = nSession + 1;
%                         nTouch = 0;
%                         pause(1)
                    end
                end
            else
                if nTrial==34
                    %                     nRect = 1;
                    nTrial = 0;
                    fprintf('Session: %d  Step: %d\n',nSession,nStep)
                    refreshTouchQueue = 1;
                    expdata(nSession).touch = touchSession;
                    %                     touchSession = nan(42,9);
                    expdata(nSession).id = rectID;
                    nSession = nSession + 1;
                    nTouch = 0;
                    if nSession>16
                        expEnd = now;
                    end
                end
            end  
            %             if nCorrectTrial>=1
            %                 rectColors = repmat(gray,nRect,1);
            %             else
            if trialNum>=trialNumSOSS||nStep==1||~testOrder
                [displayedRects,irect,rectColors] = setPattern(shiftRects,rectColor,nRect);
            end
            %             end
            patternDisplayed = 0;
            updatePattern = 0;

        end
        
        if refreshTouchQueue
            expdata(nSession).touch = touchSession(1:nTouch,:);
            expdata(nSession).id = rectID;
            saveData();
            TouchQueueStop(dev);
            TouchQueueRelease(dev);
            pause(0.2);
            TouchQueueCreate(w, dev);
            TouchQueueStart(dev);
            refreshTouchQueue = 0;
        end
            
    end
    
    TouchQueueStop(dev);
    TouchQueueRelease(dev);
    RestrictKeysForKbCheck([]);
    ShowCursor(w);
    PsychPortAudio('Close');
    sca;
    
    expdata(nSession).touch = touchSession(1:nTouch,:);
    expdata(nSession).id = rectID;
    saveData();
    fclose(fid);
catch
    TouchQueueRelease(dev);
%     RestrictKeysForKbCheck([]);
    PsychPortAudio('Close');
    sca;
    psychrethrow(psychlasterror);
end
% quit
end

% To give juice for a specific duration
function give_juice(ard, duration, nRepeat)
if ~exist('nRepeat','var') || isempty(nRepeat)
    nRepeat = 1;
end
for ii=1:nRepeat
    % Write 1 to digital pin 'which_pin' to turn the water pump on.
    ard.writeDigitalPin('D2', 1);
    % Duration of the pulse in seconds
    WaitSecs(duration);
    % Write 0 to digital pin 'which_pin' to turn the water pump on.
    ard.writeDigitalPin('D2', 0);
    WaitSecs(0.1);
end
end

function [visible_squares, visible_squares_coords] = calculate_squares (screenWidth,screenHeight,nRectX,nRectY,rectWidth,rectHeight,gapLen,xshift,yshift)
nRects = nRectX * nRectY;

% Calculate centers of square positions
x_blank = floor((screenWidth - nRectX*rectWidth - (nRectX-1)*gapLen)/2); % top left square
y_blank = floor((screenHeight - nRectY*rectHeight - (nRectY-1)*gapLen)/2);
if abs(xshift)>x_blank
    xshift=sign(xshift)*x_blank;
end
if abs(yshift)>y_blank
    yshift=sign(yshift)*x_blank;
end
x_coord_center_squares_test = x_blank+xshift+round(rectWidth/2):rectWidth+gapLen:screenWidth-x_blank+xshift;
y_coord_center_squares_test = y_blank+yshift+round(rectHeight/2):rectHeight+gapLen:screenHeight-y_blank+yshift;

% Create a vector with the psychtoolbox-style-coords of all squares and their id's.
% Id's are numbered from up to down and from left to right
rects = zeros(nRects, 5);
iRect = 0;
for x = 1 : nRectX
    for y = 1 : nRectY
        iRect = iRect + 1;
        rects(iRect, 1) = x_coord_center_squares_test(x) - rectWidth/2;
        rects(iRect, 2) = y_coord_center_squares_test(y) - rectHeight/2;
        rects(iRect, 3) = x_coord_center_squares_test(x) + rectWidth/2;
        rects(iRect, 4) = y_coord_center_squares_test(y) + rectHeight/2;
        rects(iRect, 5) = iRect;
    end
end

% Select the squares which can be lighted during the test period. In this
% case, I am selecting alternated lines and columns
square_ids_mat = reshape(1 : nRects, nRectY, nRectX);
visible_squares = square_ids_mat;
% visible_squares(2 : 2 : end, :) = [];
% visible_squares(:, 2 : 2 : end) = [];
% visible_squares([1 2 end], :) = [];
% visible_squares(:, [1 2 3 end-2 end-1 end]) = [];
visible_squares = visible_squares(1:end);

% To select the coords of the visible squares psychtoolbox-style. The 6th
% col is the new id, which takes into account only the visible squares
visible_squares_coords = [rects(visible_squares,:) (1:length(visible_squares))'];
end

% Set pattern to be displayed
function [displayedRects,irect,rectColors] = setPattern(shiftRects, rectColor,nRect)
while(1)
    irect = randsample(size(shiftRects,1),nRect);
    irect = sort(irect);
    displayedRects = shiftRects(irect,:);
    if nRect>1
        pairs = nchoosek(1:length(irect),2);
        d = zeros(size(pairs,1),1);
        for ii=1:size(pairs,1)
            d(ii) = hypot(displayedRects(pairs(ii,1),1)-displayedRects(pairs(ii,2),1),...
                displayedRects(pairs(ii,1),2)-displayedRects(pairs(ii,2),2));
        end
        if all(d>2*(displayedRects(1,3) - displayedRects(1,1)))
            break;
        end
    else
        break;
    end
end
rectColors = repmat(rectColor,nRect,1);
end

% To save a copy of the code
function copypath = copy_code(folderpath,fname)
codepath = mfilename('fullpath');
codepath = [codepath '.m'];
[~,codename,~] = fileparts(codepath);
filename = [fname '-' codename '.m'];
copypath = fullfile(folderpath,filename);
copyfile(codepath,copypath);
end

%%
function flicker_touched_target(w, rectColors, displayedRects, i_square)
black = [0 0 0]; %BlackIndex(CONFIG.main_display);
flickerRectColors = rectColors;
flickerRectColors(i_square,:) = black;
for i_flicker = 1 : 6
    Screen('FillRect', w, rectColors', displayedRects');
    Screen('FillRect', w, flickerRectColors', displayedRects');
    Screen('Flip', w);
    Screen('FillRect', w, rectColors', displayedRects');
    Screen('FillRect', w, flickerRectColors', displayedRects');
    Screen('Flip', w);
    Screen('FillRect', w, rectColors', displayedRects');
    Screen('Flip', w);
end
end

%%
function flicker_screen_punishment(w,colorB,colorF)
flickerBegin = GetSecs;
while GetSecs-flickerBegin<2
    Screen('FillRect', w, colorF); % To show a red screen
    Screen('Flip', w);
    Screen('Flip', w);
    Screen('FillRect', w, colorB);
    Screen('Flip', w);
    Screen('Flip', w);
end
end
