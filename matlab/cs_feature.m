function CAF = cs_feature(inputIQ, N, tau_max, est_len, N_hop, tau_hop)

if ~exist('N_hop', 'var')
    N_hop = 1;
end
if ~exist('tau_hop', 'var')
    tau_hop = 1;
end

index_matrix = (1:est_len)' + (0:(numel(inputIQ)-est_len));
in_matrix = inputIQ(index_matrix);
tau_vector = 0:tau_hop:tau_max;
CAF = zeros(N, numel(tau_vector));

for tau_index = 1:numel(tau_vector)
    tau = tau_vector(tau_index);

    if est_len == 1
        in_corr = in_matrix(1:end-tau) .* conj(in_matrix(1+tau:end));
    else
        in_corr = (sum(in_matrix(:, 1:end-tau) .* conj(in_matrix(:, 1+tau:end)), 1));
    end

    nFeat = numel(1 : N_hop : (numel(in_corr) - N));
    in_spec = zeros(N, nFeat);
    for i = 1:nFeat
        in_spec(:, i) = fft(in_corr(N_hop*(i-1)+1 : N_hop*(i-1)+N));
    end
    in_spec_mean = mean(abs(in_spec), 2);
    CAF(:, tau_index) = in_spec_mean;
end

end

