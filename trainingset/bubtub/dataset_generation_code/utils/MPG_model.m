function img_poisson_gaussian = MPG_model(clean_img, RPN, sigma)

% ----- Pixel Well Depth ----- %
% RPN           relative photon number for poisson noise
% sigma         sigma of gaussian noise


% ----- Image normalization ----- %
clean_img = double(clean_img);
norm_factor = 65535;  % remove outliers
clean_img = clean_img / norm_factor;

% ----- Intensity to photon number -------%
clean_img_photon = clean_img * RPN;


% ----- Shot Noise ----- %
img_poisson = poissrnd(clean_img_photon);
img_poisson = img_poisson * norm_factor / RPN;
% ---------------------- %

% ----- Read Noise ----- %
img_poisson_gaussian = int16(img_poisson + sigma*randn(size(img_poisson)));
% ---------------------- %

end

