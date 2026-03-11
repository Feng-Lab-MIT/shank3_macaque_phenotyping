function mmn_frequency
% parameters 
N_trials    =400;
 
% Amplitude 
tone_amp      = 1.5; 

% Frequency
sound_freq_1 = 500; 
sound_freq_2 = 1000;
sound_freq_3 = 2000;
sound_freq_4 = 4000;

% Duration
tone_time     = 0.05; % unit is in second
blank_time  = 0.7;
data_folder = 'C:\toolbox\data\';
Fs = 44100;  % sampling frequency 

% store data.
data.task_type     = 'MMN frequency';
data.N_trials      = N_trials;
% data.P_lo          = P_lo;
data.tone_amp      = tone_amp;
data.tone_time     = tone_time;
data.blank_time    = blank_time;
data.data_folder   = data_folder;
data.Fs            = Fs;

 %% set up
AssertOpenGL;
tonedata_1 = tone_amp*sin(sound_freq_1*2*pi*(0:1/Fs:tone_time-1/Fs));
tonedata_2 = tone_amp*sin(sound_freq_2*2*pi*(0:1/Fs:tone_time-1/Fs));
tonedata_3 = tone_amp*sin(sound_freq_3*2*pi*(0:1/Fs:tone_time-1/Fs));
tonedata_4 = tone_amp*sin(sound_freq_4*2*pi*(0:1/Fs:tone_time-1/Fs));
 InitializePsychSound;

aDataFldr = [data_folder, datestr(now,'yyyymmdd')];
if ~exist(aDataFldr, 'dir'); mkdir(aDataFldr); end

%%  set up Aruduino
port_timeStamp_1 = 2; 
port_timeStamp_2  = 4; 
port_timeStamp_3  = 8; 
port_timeStamp_4  = 16; 

%% set ports for markers
SerialPortObj=serial('COM3', 'TimeOut', 1); 
SerialPortObj.BytesAvailableFcnMode='byte'; 
SerialPortObj.BytesAvailableFcnCount=1; 
SerialPortObj.BytesAvailableFcn=@ReadCallback; 
fopen(SerialPortObj); 
fwrite(SerialPortObj, 0,'sync'); 
%% start trials
timestamps = zeros(N_trials,1);
events     = zeros(N_trials,1);
startStop_events =zeros(2,1);
tic;
startStop_events(1) = toc;

  load('stimuli.mat');
  a=stimuli; 
 % load('stimuli2.mat');
 % a=stimuli2;
 
for j=1:N_trials
    if a(j,1)==1
        tonedata = tonedata_1; lo_hi_FLG = 1; %% frq1
        port_timeStamp = port_timeStamp_1;
    elseif a(j,1)==2
        tonedata = tonedata_2; lo_hi_FLG = 2; %% frq2
        port_timeStamp = port_timeStamp_2;
    elseif a(j,1)==3
        tonedata = tonedata_3; lo_hi_FLG = 3; %% frq3
        port_timeStamp = port_timeStamp_3;
    else
        tonedata = tonedata_4; lo_hi_FLG = 4; %% frq4
        port_timeStamp = port_timeStamp_4;

    end
    
    pahandle = PsychPortAudio('Open', [], [], 0, Fs, 1);
    PsychPortAudio('FillBuffer', pahandle, tonedata);
    PsychPortAudio('Start', pahandle, 1);
     fwrite(SerialPortObj, port_timeStamp,'sync');
     pause(0.005); 
     fwrite(SerialPortObj, 0,'sync'); 
    PsychPortAudio('Stop', pahandle);
    PsychPortAudio('Close', pahandle);

   
    timestamps(j) = toc;
    events(j)     = lo_hi_FLG;

  pause(blank_time);

end
startStop_events(2) = toc;
fclose(SerialPortObj); 
delete(SerialPortObj); 
clear SerialPortObj;

%% store data
data.timestamps = timestamps;
data.events     = events;
data.startStop_events =startStop_events;
aFileName = [aDataFldr, '\data', datestr(now,'_yyyymmdd_HHMMSS')];
save(aFileName, 'data');
 fprintf('完成一个循环\n');

