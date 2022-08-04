function filt_bold = filter_bold(bold,flp,fhp,tr)
    TR=2;  % Repetition Time (seconds)
    
    % Bandpass filter settings
    fnq=1/(2*TR);                 % Nyquist frequency
    flp = 0.008;                    % lowpass frequency of filter (Hz)
    fhi = 0.08;                    % highpass
    Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
    k=2;                          % 2nd order butterworth filter
    % NOTE: Default is lowpass, should be bandpass. Still works
    [bfilt,afilt]=butter(k,Wn);   % construct the filter
    N = size(bold, 1);
    for seed=1:N
        x(seed,:)=detrend(bold(seed,:, :)-mean(bold(seed,:,:)));
        filt_bold(seed,:)=filtfilt(bfilt,afilt,x(seed,:));        
    end
