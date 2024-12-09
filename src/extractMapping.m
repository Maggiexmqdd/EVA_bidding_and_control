function extractMapping(sol_mpt,tau)
% Retrieve the explicit solution for each critical region
regions = sol_mpt.xopt.Set; % Critical regions
num_regions = length(regions);

% extract optimal value function-- affine mapping for x = F * theta + g
% primal refers to the primal solution
for j = 1:num_regions
    mapping.F = regions(j).Functions('primal').F; % Coefficients of theta
    mapping.g = regions(j).Functions('primal').g; % Constant offset

    % Get the region constraints (E * theta <= f)
    mapping.region.E = regions(j).A;
    mapping.region.f = regions(j).b;

    % Save the mapping to a file
    filename = sprintf('%smapping_%d_%d.mat', outputPath, tau, j - 1);
    save(filename, 'mapping');
end
end