

% Loop through plp_0.mat to plp_23.mat
for i = 10
    % Construct the filename dynamically based on the loop index
    filename = sprintf('../output/crs/plp_%d.mat', i);
    load(filename)

    num_var  = size(A,2);
    num_para = size(At,2);

    x     = sdpvar(num_var, 1);
    theta = sdpvar(num_para, 1);

    Constraints = [];
    Constraints = [Constraints;
        A * x <= b;
    ];
    Constraints = [Constraints;
        Aeq * x  == beq + Feq * theta;
    ];
    At=double(At);
    bt=double(bt);
    Constraints = [Constraints;
        At * theta <= bt;
    ];
    Constraints = [Constraints;
        -1 <= theta <= 1;
    ];

    Objective = c' * x;

    plp = Opt(Constraints, Objective, theta, x);
    sol_mpt = plp.solve();

    VF = sol_mpt.xopt.Set.getFunction('obj');
    crs = sol_mpt.xopt.Set;

    num_cr = length(VF);

    for j = 1:num_cr
        name = sprintf('cr%d_%d', i, j-1);
        cr.vf_coeff_t = VF(j).F;
        cr.vf_b = VF(j).g;
        cr.E = crs(j).A;
        cr.f = crs(j).b;
        save(['../output/crs_80/', name, '.mat'], "cr");
    end
    
    % 提取 position 8 和 position 16 的数据
    figure;
    sol_mpt.xopt.fplot('primal', 'position', 8);  % 提取 position 8 的数据
    xlabel('s');
    ylabel('p^{ch}_i(s)');

    figure;
    sol_mpt.xopt.fplot('primal', 'position', 16); % 提取 position 16 的数据
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
end

disp('All files processed successfully.')
