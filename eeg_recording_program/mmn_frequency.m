function mmn_frequency
% parameters 
N_trials    =400;
P_lo        = 0.8;  
 
% Amplitude  
tone_amp      = 1.5; 
% Frequency
sound_freq_lo = 600; 
sound_freq_hi = 1200;
% Duration
tone_time     = 0.05; % unit is in  second
blank_time  = 0.7;
data_folder = 'C:\toolbox\data\';
Fs = 44100;  % sampling frequency 

% store data.
data.task_type     = 'MMN frequency';
data.N_trials      = N_trials;
data.P_lo          = P_lo;
data.sound_freq_lo = sound_freq_lo;
data.sound_freq_hi = sound_freq_hi;
data.tone_amp      = tone_amp;
data.tone_time     = tone_time;
data.blank_time    = blank_time;
data.data_folder   = data_folder;
data.Fs            = Fs;



%% set up
AssertOpenGL;
tonedata_hi = tone_amp*sin(sound_freq_hi*2*pi*(0:1/Fs:tone_time-1/Fs));
tonedata_lo = tone_amp*sin(sound_freq_lo*2*pi*(0:1/Fs:tone_time-1/Fs));
 InitializePsychSound;

aDataFldr = [data_folder, datestr(now,'yyyymmdd')];
if ~exist(aDataFldr, 'dir'); mkdir(aDataFldr); end

%%  set up Aruduino
port_timeStamp_freq = 2; % Trigger for Standard
port_timeStamp_odd  = 4; % Trigger for Oddball

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

for j=1:N_trials

    if rand <= P_lo
        tonedata = tonedata_lo; lo_hi_FLG = 0; %% frq
        port_timeStamp = port_timeStamp_freq;

    else
        tonedata = tonedata_hi; lo_hi_FLG = 1; %% odd
        port_timeStamp = port_timeStamp_odd;

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

