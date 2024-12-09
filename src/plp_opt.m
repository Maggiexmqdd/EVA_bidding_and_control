function num_cr = plp_opt(t)
% This code requires the Multi-Parametric Toolbox
% (http://control.ee.ethz.ch/~mpt/).

filename = sprintf('../output/crs/plp_%d.mat', t);
load(filename)


% Define problem dimensions
num_var  = size(A,2);
num_para = size(At,2);
At=double(At);
bt=double(bt);

% Define decision variables
x     = sdpvar(num_var, 1);
theta = sdpvar(num_para, 1);

% Define constraints
Constraints = [];
Constraints = [Constraints;A * x <= b]; % Inequality constraints
Constraints = [Constraints;Aeq * x  == beq + Feq * theta];% Equality constraints
% Constraints = [Constraints;At * theta <= bt];
Constraints = [Constraints; -1 <= theta <= 1]; % Box constraints on theta

% Define objective function
Objective = c' * x;

% Solve the parametric linear programming problem
plp = Opt(Constraints, Objective, theta, x);
sol_mpt = plp.solve();
modelname=sprintf('model_%d', t);
save(['../input/', modelname, '.mat'], "sol_mpt");
% Since we have an explicit solution, we can plot the partition
sol_mpt.xopt.plot

% Extract CRs and value functions
crs = sol_mpt.xopt.Set; % Critical regions
VF = sol_mpt.xopt.Set.getFunction('obj');


% Save the coefficients and constraints of each critical region
num_cr = length(VF);
for j = 1:num_cr
    name = sprintf('cr%d_%d', 1i, j-1);
    cr.vf_coeff_t = VF(j).F;
    cr.vf_b = VF(j).g;
    cr.E = crs(j).A;
    cr.f = crs(j).b;
    save(['../output/crs_80/', name, '.mat'], "cr");
end

% 提取optimal decision的数据, x(p_ch) 和  x+num_var(p_dis) 的数据
pos=8;
figure;
sol_mpt.xopt.fplot('primal', 'position', pos);  % 提取 position 8 的数据
xlabel('s');
ylabel('p^{ch}_i(s)');
pos2=pos+num_var;
figure;
sol_mpt.xopt.fplot('primal', 'position', pos2); % 提取 position 16 的数据
xlabel('t');
ylabel('p^{dis}_i(s)');
% 
% % 将 position 16 数据转换为相反数
% data_pos16.Values = -data_pos16.Values;
% 
% % 绘图
% figure;
% hold on; % 保持当前图形，使后续绘图叠加在同一张图上
% 
% % 绘制 position 8 和相反数后的 position 16 曲线
% plot(data_pos8.Values, 'DisplayName', 'Position 8');
% plot(data_pos16.Values, 'DisplayName', 'Position -16');
% 
% % 添加图例标签
% legend('Position 8', 'Position -16');
% 
% % 设置轴标签
% xlabel('s');
% ylabel(sprintf('p_%d (s)', i));
% 
% disp(['Finished processing: ', filename]);
% hold off; % 结束叠加

% end
disp(['Finished processing: ', filename])
disp('All files processed successfully.')
end

