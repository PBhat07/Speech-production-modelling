close all;
clear all;

[sig1, fs1]= audioread('hid_f.wav');
[sig2, fs2]= audioread('hid_m.wav');
fprintf('Signal duration: %.2f seconds\n', length(sig1) / fs1);
fprintf('Signal duration: %.2f seconds\n', length(sig2) / fs2);
segmentlen = 100;
noverlap = 90;
NFFT = 128;
%Spectrogram of the signal
%spectrogram(sig1,segmentlen,noverlap,NFFT,fs1,'yaxis')
%spectrogram(sig2,segmentlen,noverlap,NFFT,fs2,'yaxis')

%Segmentation of signal
x = sig1(round(0.024 * fs1):round((0.024+0.1) *fs1));
y = sig2(round(0.025 * fs2):round((0.025+0.1) *fs2));


%Estimate lpc coefficient
A = lpc(x,20);
B = lpc(y,20);
rts1 = roots(A);
rts1 = rts1(imag(rts1)>0);
rts2 = roots(B);
rts2 = rts2(imag(rts2)>0);
angz1 = atan2(imag(rts1),real(rts1));
angz2 = atan2(imag(rts2),real(rts2));
[frqs1,indices1] = sort(angz1.*(fs1 / (2*pi)));
for i=1:3
    fprintf('formant %d:%.2f\n',i,frqs1(i))
end    

[frqs2,indices2] = sort(angz2.*(fs2 / (2*pi)));
for i=1:3
    fprintf('formant %d:%.2f\n',i,frqs2(i))
end  

num1=1;
num2=1;


spectrum1 = fft(x);
spectrum2 = fft(y);

% Frequency vector for plotting the spectrum
f_vec1 = linspace(0, fs1/2, length(spectrum1)/2);
f_vec2 = linspace(0, fs2/2, length(spectrum2)/2);

% Calculate the amplitude spectrum (in dB)
amplitudeSpectrum1 = 10 * log10(abs(spectrum1));
amplitudeSpectrum2 = 10 * log10(abs(spectrum2));

% Compute the LPC filter response
num = 1;
den1 = A; 
den2 = B;

[h1, f1] = freqz(num, den1,512, fs1);
[h2, f2] = freqz(num, den2,512, fs2);

% Mean fundamental frequency
f01 = pitch(sig1,fs1);
f02 = pitch(sig2,fs2);
fm1=mean(f01);
fm2=mean(f02);

% Define parameters
Fs = 24000;          % Sampling rate
duration = 1;      % Duration of the excitation signal in seconds
F01 = fm1;   % Fundamental frequency (adjust as needed)
F02 = fm2;

% Calculate the number of samples
num_samples = round(Fs * duration);

% Calculate the interval between impulses
impulse_interval1 = round(Fs / F01);
impulse_interval2 = round(Fs / F02);

% Initialize the impulse train signal
impulse_train1 = zeros(1, num_samples);
impulse_train2 = zeros(1, num_samples);

% Set impulse amplitudes to 1 at appropriate intervals
impulse_train1(1:impulse_interval1:end) = 1;
impulse_train2(1:impulse_interval2:end) = 1;




% Filter the impulse train using the LPC filter
filtered_signal1 = filter(1, A, impulse_train1);
filtered_signal1 = filtered_signal1 / max(abs(filtered_signal1));

filtered_signal2 = filter(1, A, impulse_train2);
filtered_signal2 = filtered_signal2 / max(abs(filtered_signal2));

% Play the filtered signal after sementation
filtered_signal3=filtered_signal1(1:0.1*Fs);
filtered_signal4=filtered_signal2(1:0.1*Fs);

% Use the audiowrite function to save the filtered signal to a .wav file
filename1 = 'hid_fout2.wav';
filename2 = 'hid_mout2.wav';
filename3 = 'hid_fout3.wav';
filename4 = 'hid_mout4.wav';

audiowrite(filename1, filtered_signal1, Fs);
audiowrite(filename2, filtered_signal2, Fs);

%segment the output sound
audiowrite(filename3, filtered_signal3,Fs);
audiowrite(filename4, filtered_signal4, Fs);


% plot (the amplitude spectrum in dB);
figure('Name', 'Figure 3');
subplot(2,1,1);
hold on;
plot(f_vec1, amplitudeSpectrum1(1:length(f_vec1)),'r');
plot(f1,10*log10(abs(h1)),'b');
hold off;
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
title('Amplitude Spectrum and LPC filtere response of Speech Segment');

subplot(2,1,2);
hold on;
plot(f_vec2, amplitudeSpectrum2(1:length(f_vec2)),'r');
plot(f2,10*log10(abs(h2)),'b');
hold off;
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
title('Amplitude Spectrum of Speech Segment');

%Plot the original impulse train and the filtered signal
figure('Name','figure 6 ');
t = (0:num_samples-1) / Fs;
subplot(2, 1, 1);
stem(t, impulse_train1, 'marker', 'o', 'lineWidth', 1.5);
title('Original Impulse Train');
grid on;

subplot(2, 1, 2);
plot(t, filtered_signal2);
title('Filtered Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

figure('Name','figure 7 ');
t = (0:num_samples-1) / Fs;
subplot(2, 1, 1);
stem(t, impulse_train2, 'marker', 'o', 'lineWidth', 1.5);
title('Original Impulse Train');
grid on;

subplot(2, 1, 2);
plot(t, filtered_signal2);
title('Filtered Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;







