function touchRectTrainerV2(setupID,monkeyID)
% A simple task in which the subject has to touch a central gray square in
% the screen to % receive reward (= juice). Multitouch was disabled for %`0
% this demo.The touch sensitivity of the mini-screen in the back of the box
% is disabled while while the demo is running. Press 'ESC' to exit the
% demo.
%
% _________________________________________________________________________
%
% History:
% 22-Apirl-2021 Modified from touchMultiRect3.
% 21-Jun-2022 modified from touchRectTrainer4zhen. 
% 27-Jul-2022 add total_trials
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
            case 1%{1,2}
                setupType=5;
            otherwise
                setupType=6;
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
    case 1% sculptor with surface pro7
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
    case 5 % sculptor with NUC10
        CONFIG.main_display_area = [1024 0 2944 1080];
        touchscreenID = 'Silicon Works Multi-touch SW4101C';
        pixel2mm = 165/1080;
    case 6 % weichensi with NUC10
        CONFIG.main_display_area = [1024 0 2944 1080];
        touchscreenID = 'WingCool Inc. TouchScreen';
        pixel2mm = 165/1080;
end
water_time = 0.06;
fig_path = fullfile(pwd,'apple01.png');
fig = imread(fig_path);
data_dir = fullfile(pwd,'Results');
% Number of the surface screen
CONFIG.main_display = max(Screen('Screens'));
CONFIG.nRect = 1;
CONFIG.nRep = 1;%5;
CONFIG.randLoc = true;
CONFIG.randColor = false;
CONFIG.rectColor = [0 0 1;0 1 0;1 1 0;1 0 1;0 1 1];
CONFIG.flicker = true;%false;%true;
CONFIG.beepon = false;
CONFIG.trialInterval = 1;
CONFIG.pre = false;%true;
CONFIG.total_trials = 750;
nRect = CONFIG.nRect;
nRep = CONFIG.nRep;
randLoc = CONFIG.randLoc;
randColor = CONFIG.randColor;
rectColor = CONFIG.rectColor;
flicker = CONFIG.flicker;
beepon = CONFIG.beepon;
total_trials = CONFIG.total_trials;
colorNum = size(rectColor,1);

dev = GetTouchDeviceIndices([], 1, touchscreenID);



% Find the color values which correspond to white and black: Usually
% black is always 0 and white 255, but this rule is not true if one of
% the high precision framebuffer modes is enabled via the
% PsychImaging() commmand, so we query the true values via the
% functions WhiteIndex and BlackIndex:
white=WhiteIndex(CONFIG.main_display);
black=BlackIndex(CONFIG.main_display);
gray=(white + black) / 2;
red = [1.0 0 0];
cyan = [0 1.0 1.0];
green = [0 1.0 0];
orange = [255 127 0]/255;

% Open a default onscreen window with black background color and 0-1 color
% range:
[w, rect] = PsychImaging('OpenWindow', CONFIG.main_display, black, CONFIG.main_display_area);
blackBG = true;
textureIndex = Screen('MakeTexture', w, fig);
% textureIndex2 = Screen('MakeTexture', w, fig2);  

% Get the center coordinate of the window in pixels
[xCenter, yCenter] = RectCenter(rect);

% Get the size of the on screen window in pixels
% For help see: Screen WindowSize?
[screenWidth, screenHeight] = Screen('WindowSize', w);

% Define error audio
if beepon
    Fs = 44100;
    pahandle = PsychPortAudio('Open', 8, [], 2, Fs);
    beepWaveform = MakeBeep(1000, 0.5, Fs);
    beepWaveform = repmat(beepWaveform,2,1);
    PsychPortAudio('FillBuffer', pahandle, beepWaveform);
end

% Make a base Rect of 600 by 600 pixels. This is the rect which defines the
% size of our square in pixels. Rects are rectangles, so the
% sides do not have to be the same length. The coordinates define the top
% left and bottom right coordinates of our rect [top-left-x top-left-y
% bottom-right-x bottom-right-y]. The easiest thing to do is set the first
% two coordinates to 0, then the last two numbers define the length of the
% rect in X and Y. The next line of code then centers the rect on a
% particular location of the screen.
% The side length of a square is 20 mm
rectWidth = round(20/pixel2mm); 
rectHeight = round(20/pixel2mm);
gapLen = round(3/pixel2mm); % The gap between two adjacent rectangles is 3 mm

baseRect = [0 0 rectWidth rectHeight];

xshift = 0;%gapLen + rectWidth;
yshift = 0;%-gapLen - rectHeight;
[visible_squares, visible_squares_coords] = calculate_squares (screenWidth,...
    screenHeight,6,4,rectWidth,gapLen,xshift,yshift);    
allRects = visible_squares_coords(:,1:4);

% % pkg load statistics
% rand("seed",100*sum(clock))
rng(sum(100*clock));


% Set the color of our square to full red. Color is defined by red green
% and blue components (RGB). So we have three numbers which
% define our RGB values. The maximum number for each is 1 and the minimum
% 0. So, "full red" is [1 0 0]. "Full green" [0 1 0] and "full blue" [0 0
% 1]. Play around with these numbers and see the result.
%rectColor = gray;
fps = Screen('NominalFrameRate',w);


% Get maximum supported dot diameter for smooth dots:a
[~, maxSmoothPointSize] = Screen('DrawDots', w);

% Select good diameter for touch point blobs, but no more than what
% 'DrawDots' supports:
baseSize = min(RectWidth(rect) / 20, maxSmoothPointSize);

%% Function to process touch events
    function process_touch_event()
        
        % TouchEventAvail reports the number of events in a touch queue.
        % One single touch can contain many events.
        while TouchEventAvail(dev)
            
            evt_count = evt_count + 1;
            
            %  Return oldest pending event
            evt = TouchEventGet(dev, w);
            
            nTouch = nTouch + 1;
            tdata(nTouch,:) = [evt.MappedX, evt.MappedY evt.Time evt.Type];
            
            % Touch blob id - Unique in the session at least as
            % long as the finger stays on the screen:
            id = evt.Keycode;
            % fprintf("%d %d\n", evt.Keycode, evt.Type)
            
            % Only consider the id of the firstard = serialport('/dev/ttyACM0', 9600); touch event
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
                    Screen('DrawDots', w, [evt.MappedX; evt.MappedY], ...
                        baseSize, [1,1,1], [], 1, 1);
                    
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
        
        % To determine if blobcol is empty
        if evt_count && isempty(blob)
            evt_count = 0;
        end
        
        if buttonstate
            Screen('FrameRect', w, [1, 1, 0], [], 5);
        end
        
    end

%% Function to save task data
    function save_data()
        timeInfo(3,:) = clock;
        expData = expdata(1:trialNum-1);
        tRight = tright(1:nRight);
        tWrong = twrong(1:nWrong);

%         allRects = visible_squares_coords(:,1:4);
        save(filepath ,'expData','timeInfo','allRects','monkeyID','tRight','tWrong','timer_exp_start')
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
    function initialize_para()
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
        
        if randLoc
            irect = randsample(length(visible_squares),1);
%             icorrect = 1;
%             irect = randsample(length(visible_squares),nRect);
%             irect = sort(irect);
%             icorrect = randsample(nRect,1);
            displayedRect = allRects(irect,:);           
        else
%             icorrect = 1;
            irect = 1;
            displayedRect = CenterRectOnPointd(baseRect, xCenter, yCenter);
            allRects = displayedRect;
%         else
%             irect = randsample(6,nRect);
%             irect = sort(irect);
%             icorrect = randsample(nRect,1);
%             displayedRects = centeredRects(irect,:);
        end
        
        clear expdata
        expdata.tdata = nan(100,4);
        expdata.rect = nan(nRect,4);
        touchEvents = nan(nRep*nRect,10);
        expdata.touch = touchEvents;
%         expdata.id = nan;
        expdata = repmat(expdata,100,1);
        expdata(1).rect = displayedRect;
%         expdata(1).id = icorrect;
        timeInfo = zeros(3,6);
        firstTouch = true;
%         timeInfo(1,:) = tclock;
        
        trialNum = 1;
        repNum = 0;
        touchNum = 0;
        nTouch = 0;
        tright = nan(2000,1);
        twrong = nan(2000,1);
        nRight = 0;
        nWrong = 0;
        consecutive = 0;
        if randColor
            indColor = randsample(colorNum,1);
        else
            indColor = 1;
        end
        
        evt_count = 0;
        first_event_id = [];
        buttonstate = 0;
        blob = [];
        monkey_touching_screen = 0;
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
    evt_count = 0;
    first_event_id = [];
    buttonstate = 0;
    blob = [];
    monkey_touching_screen = 0;
    trialEnd = GetSecs;
    tdata = nan(100,4);
    
    tclock = [];
    expBegin = [];
    expEnd = [];
    nextDay = [];
    checkPoint = [];
    folderpath = [];
    fname = [];
    filepath = [];
    
%     escKey = KbName('ESCAPE');
%     resetKey = KbName('SPACE');
    initialize_para;
    timer_exp_start = Screen('Flip', w);
    % To save a copy of the code
    copypath = copy_code(folderpath,fname);

    t0 = tic;
    % Main loop: Run until keypress:
    escape_pressed = 0;
    while ~escape_pressed
        escape_pressed = KbCheck;
%         [~, ~, keyCode] = KbCheck;
%         if keyCode(escKey)
%             break;
%         end
%        if keyCode(resetKey)
%            trialNum = 1;
%            repNum = 0;
%            touchNum = 0;
%            sprintf('touchNum reset at %s',datestr(now))
%            pause(0.1);
%        end
        tnow = now;
        if tnow>nextDay            
%             save_data;
            initialize_para;
            copypath = copy_code(folderpath,fname);          
        end
        
        if tnow>expEnd
%             break;
            save_data;
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
%             Screen('FillRect', w, black);
%             Screen('DrawTexture', w, textureIndex);
            timer_exp_start = Screen('Flip', w);
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
        if toc(t0)>60
            fprintf('%s\n',datestr(now))
            t0 = tic;
        end
        
        process_touch_event;
        
        if firstTouch&&~isempty(blob)
            timeInfo(2,:) = clock;
            firstTouch = false;
        end

        % Don't start trial if the monkey keeps touching the screen
        while trialEnd&&GetSecs-trialEnd<=CONFIG.trialInterval
            while monkey_touching_screen
%                 process_touch_event;%
                TouchEventFlush(dev);
                blob = [];
                monkey_touching_screen = 0;
                WaitSecs(1);
                process_touch_event;%
                escape_pressed = KbCheck;
                if escape_pressed
                    break;
                end
            end
        end
        
        % Draw the rectangle
        if ~CONFIG.pre
            Screen('FillRect', w, rectColor(indColor,:), displayedRect');
        else
            Screen('DrawTexture', w, textureIndex,[],displayedRect);
        end
        %         Screen('DrawText', w, sprintf('The %d-th trial (%d/%d)',trialNum,repNum,nRep), 80, 45, [255 0 0]);
        Screen('Flip', w);
        
        while isempty(blob)&&~escape_pressed&&now<expEnd
            escape_pressed = KbCheck;
            process_touch_event;
        end
        if ~isempty(blob)
            if IsInRect(blob.x, blob.y, displayedRect)
                blob_inside_rect = 1;
            else
                blob_inside_rect = 0;
            end
            
            t = datestr(now);
            touchNum = touchNum + 1;
            touchEvents(touchNum,:) = [blob.x blob.y datevec(t) irect blob_inside_rect];
            blob = [];
            TouchEventFlush(dev);
        else
            blob_inside_rect = -1;
        end
        

        % Done repainting - Show it:
        Screen('FillRect', w, black);
%         Screen('DrawTexture', w, textureIndex,[],displayedRect);
        Screen('Flip', w);
        
        switch blob_inside_rect 
            case 1
                if ~isempty(ard)
                    give_juice(ard, water_time, 1);
                else
                    fprintf('Correct\n')
                end
                fprintf('Water Delivered at %s\n',t)
                fprintf('The %d-th trial (%d/%d)\n',trialNum,repNum,nRep)
                nRight = nRight + 1;
                tright(nRight) = etime(clock,tclock);
                fprintf('The hit rate is %f\n',nRight/(nRight+nWrong))
                
                %             repNum = repNum + 1;
                consecutive = consecutive + 1;
                if consecutive==5
                    if nRight>520
                        %                     give_juice(a, 0.005, 1);
                    end
                    consecutive = 0;
                end
            case 0
                if beepon
                    PsychPortAudio('Start', pahandle, 1);
                end
                if flicker
                    flicker_screen_punishment(w,black,red);
                end
                nWrong = nWrong + 1;
                twrong(nWrong) = etime(clock,tclock);
                fprintf('The hit rate is %f\n',nRight/(nRight+nWrong))
            case -1
                continue;
        end
        trialEnd = GetSecs;
        
        repNum = repNum + 1;        
        
        if repNum == nRep
            expdata(trialNum).touch = touchEvents;
            expdata(trialNum).tdata = tdata;
            tdata = nan(100,4);
            nTouch = 0;
            repNum = 0;
            touchNum = 0;
            if randColor
                indColor = randsample(colorNum,1);
            else
                indColor = 1;
            end
            if randLoc
                irect = randsample(length(visible_squares),1);
%                 icorrect = 1;
                displayedRect = allRects(irect,:);
            end
            touchEvents = nan(nRep*nRect,10);
            %                 fprintf("\nTrials completed: %d\n", trialNum);
            trialNum = trialNum + 1;
            if trialNum>total_trials
                expEnd = now;
            else
                expdata(trialNum).rect = displayedRect;
            end
            %                expdata(trialNum).touch = touchEvents;
        end    
                
        if ~mod(trialNum, 30)
            save_data;
            TouchQueueStop(dev);
            TouchQueueRelease(dev);
            WaitSecs(0.2);
            TouchQueueCreate(w, dev);
            TouchQueueStart(dev);
            monkey_touching_screen = 0;
        end
    end
   
    TouchQueueStop(dev);
    TouchQueueRelease(dev);
    RestrictKeysForKbCheck([]);
    ShowCursor(w);
    PsychPortAudio('Close');
    sca;
   
    expdata(trialNum).touch = touchEvents;
    expdata(trialNum).tdata = tdata;
    save_data;
catch
    TouchQueueRelease(dev);
%     RestrictKeysForKbCheck([]);
    PsychPortAudio('Close');
    sca;
    psychrethrow(psychlasterror);
end

end

% To give juice for a specific duration
function give_juice(a, duration, nRepeat)
if ~exist('nRepeat','var') || isempty(nRepeat)
    nRepeat = 1;
end
for ii=1:nRepeat
    % Write 1 to digital pin 'which_pin' to turn the water pump on.
    a.writeDigitalPin('D2', 1);
    % Duration of the pulse in seconds
    WaitSecs(duration);
    % Write 0 to digital pin 'which_pin' to turn the water pump on.
    a.writeDigitalPin('D2', 0);
    WaitSecs(0.1);
end
end

function [visible_squares, visible_squares_coords] = calculate_squares (screenWidth,screenHeight,n_squares_x,n_squares_y,squareLen, gapLen,xshift,yshift)
n_squares = n_squares_x * n_squares_y;

% Calculate centers of square positions
x_blank = floor((screenWidth - n_squares_x*squareLen - (n_squares_x-1)*gapLen)/2); % top left square
y_blank = floor((screenHeight - n_squares_y*squareLen - (n_squares_y-1)*gapLen)/2);
if abs(xshift)>x_blank
    xshift=sign(xshift)*x_blank;
end
if abs(yshift)>y_blank
    yshift=sign(yshift)*x_blank;
end
x_coord_center_squares_test = x_blank+xshift+round(squareLen/2):squareLen+gapLen:screenWidth-x_blank+xshift;
y_coord_center_squares_test = y_blank+yshift+round(squareLen/2):squareLen+gapLen:screenHeight-y_blank+yshift;

% Create a vector with the psychtoolbox-style-coords of all squares and their id's.
% Id's are numbered from up to down and from left to right
squares = zeros(n_squares, 5);
square_id = 0;
for x = 1 : n_squares_x
    for y = 1 : n_squares_y
        square_id = square_id + 1;
        squares(square_id, 1) = x_coord_center_squares_test(x) - squareLen/2;
        squares(square_id, 2) = y_coord_center_squares_test(y) - squareLen/2;
        squares(square_id, 3) = x_coord_center_squares_test(x) + squareLen/2;
        squares(square_id, 4) = y_coord_center_squares_test(y) + squareLen/2;
        squares(square_id, 5) = square_id;
    end
end

% Select the squares which can be lighted during the test period. In this
% case, I am selecting alternated lines and columns
square_ids_mat = reshape(1 : n_squares, n_squares_y, n_squares_x);
visible_squares = square_ids_mat;
% visible_squares(2 : 2 : end, :) = [];
% visible_squares(:, 2 : 2 : end) = [];
% visible_squares([1 2 end], :) = [];
% visible_squares(:, [1 2 3 end-2 end-1 end]) = [];
visible_squares = visible_squares(1:end);

% To select the coords of the visible squares psychtoolbox-style. The 6th
% col is the new id, which takes into account only the visible squares
visible_squares_coords = [squares(visible_squares,:) (1:length(visible_squares))'];
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