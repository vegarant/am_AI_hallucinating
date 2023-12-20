clear('all') ; close('all');

% Create destination for the plots
dest = 'plots';
if (exist(dest) ~= 7) 
    mkdir(dest);
end
dwtmode('per', 'nodisp'); % Periodic boundary conditions for the DWT

vm = 4;                                     % Number of vanishing moments
subsampling_rate = 0.05;                    % ∈ [0,1]
sigma = 0.01;                               % min ||x||_1 s.t. ||Ax-b||_2 ≤ sigma
N = 256; % Image resolution N x N

% Sampling pattern parameters
a = 1.75;
nbr_levels = 50;
r0 = 0;

% Modify the path to the data and sampling patterns
data_path = '/mn/sarpanitu/ansatte-u4/vegarant/software/book_chapter/rec_different_sampling_rates/plots/val/splits';
samp_patt_path = '/mn/sarpanitu/ansatte-u4/vegarant/software/book_chapter/rec_different_sampling_rates/samp_patt';

im_nbr = 2;     % Modify this number according to your own preferences
model_nbr = 84; % Modify this number according to your own training

fname_core = sprintf('mod_%03d_val_im_nbr_%03d_orig.png', model_nbr, im_nbr);
fname_data = fullfile(data_path, fname_core); 

X = double(imread(fname_data))/255; % Scale image to [0,1]

subsampling_rates = [0.01, 0.05, 0.10, 0.20];

for i = 1:length(subsampling_rates)
    srate = subsampling_rates(i);
    nbr_samples = round(srate*N*N);
    fprintf('Computing sampling rate %g\n', srate);
    
    fname_patt = sprintf('spf2_cgauss_N_%d_srate_%02d_r_%d_r0_%d_a_%d.png', N, round(100*srate), nbr_levels, r0, round(100*a));
    Z = imread(fullfile(samp_patt_path, fname_patt));
    idx = find(Z);
    
    % Fourier wavelet
    fname = sprintf('aFourier_comp_patt_N_%d_srate_%g_db%d', N, srate*100, vm);
    [im_rec, wcoeff] = cil_sample_fourier_wavelet(X, sigma, idx, fullfile(dest, fname), vm);

end


