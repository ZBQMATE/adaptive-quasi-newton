function adaqn(wdr, nhid, niter, lr, mr, es, mbs, acp, sc, mfc)
%weight decay rate i.e 0, number of hid units i.e 7, number of iterations i.e 450, learnning rate i.e 0.35, momentum rate i.e 0.9, boolean whether early stopping i.e false, mini batch size i.e 20, acceptation rate i.e 1.01, sufficient curvative rate i.e 0.0001, maxium fisher container i.e 100
	warning('error', 'Octave:broadcast');
	if exist('page_output_immediately'), page_output_immediately(1); end %control octave to send its output to pagers immediately
	more off; % turn paginition off
	
	md = createmodel(nhid);
	loadfile = load('data.mat');
	datapool = loadfile.data;
	ntc = size(datapool.test.inputs, 2); %number of training cases
	
	vct = md_to_vct(md);
	ms = vct * 0; %momentum speed vector
	training_data_losses = [];
	validation_data_losses = [];
	
	if es,
		best_cur.vct = -1;
		best_cur.validation_loss = inf;
		best_cur.after_niter = -1;
	end
	
	
	
	
	
	
	t = 0; % floor(niter/mbs)
	sss = [];
	yyy = [];
	speer_grad_temp = d_loss_by_d_model(md, datapool.test, wdr);
	speer_grad = speer_grad_temp.hid_con;
	inih = diag(1./sqrt(speer_grad.^2 + sc));
	hhhggg = (1./sqrt(speer_grad.^2 + sc));
	for cur_iter = 1 : niter,
		md = vct_to_md(vct);
		tsp = mod((cur_iter - 1) * mbs, ntc) + 1; % training batch start point, easy to comprehend if ci*mbs is less than ntc
		training_batch.inputs = datapool.test.inputs(:, tsp : tsp + mbs - 1);
		training_batch.targets = datapool.test.targets(:, tsp : tsp + mbs - 1);
		
		peer_units = vct(266 * nhid + 1 : size(vct, 1));
		peer_units = peer_units - lr * hhhggg;
		
		
		
		
		%gradient = md_to_vct(d_loss_by_d_model(md, training_batch, wdr));
		
		peer_grad_temp = d_loss_by_d_model(md, training_batch, wdr);
		peer_grad = peer_grad_temp.hid_con; %this is stochastic gradient <number of hidden>*1
		
		if cur_iter == 1,
			fifo_contain = peer_grad;
		else
			fifo_contain = [fifo_contain, peer_grad]; %///6/// store grad for computing fisher imformation matrix <number of hidden> * <number of iterations>
		end
		
		if size(fifo_contain, 2) > mfc,
			fifo_contain(:, 1) = [];
		end
		
		if cur_iter == 1, %///7///
			peer_units_sum = peer_units;
		else
			peer_units_sum = peer_units_sum + peer_units;
		end
		
		
		ggg = peer_grad;
		for i = size(sss, 2):-1:1,
			sb(i) = 1/(yyy(:, i)' * sss(:, i));
			db(i) = sb(i)*sss(:, i)'*ggg;
			ggg = ggg - db(i)*yyy(:, i);
		end
		rrr = inih*ggg;
		
		for j = 1:size(sss, 2),
			beta = sb(j)*yyy(:, j)' * rrr;
			rrr = rrr + sss(:, j)*(db(j) - beta);
		end
		
		hhhggg = rrr;
		
		
		if mod(cur_iter, mbs) == 0,
			peer_ave = peer_units_sum ./ mbs;
			peer_units_sum = peer_units_sum .* 0;
			
			if t == 0,
				peer_old = peer_ave;
			else
				mdave = md;
				mdave.hid_con = reshape(peer_ave, nhid, 1);
				mdold = md;
				mdold.hid_con = reshape(peer_old, nhid, 1);
				if loss(mdave, training_batch, wdr) > acp * loss(mdold, training_batch, wdr),
					% clear L BFGS memory and fisher container
					sss = [];
					yyy = [];
					fifo_contain = [];
					peer_units = peer_old;
					continue;
				end
				
				s = peer_ave - peer_old;
				y = fifo_contain * (fifo_contain' * s);
				if dot(s,y) > sc*dot(s,s),
					sss = [sss,s];
					yyy = [yyy,y];
					peer_old = peer_ave;
				end
			
			end
			t = t + 1;
		end
		
		
		
		
		
		gradient = md_to_vct(d_loss_by_d_model(md, training_batch, wdr));
		ms = ms * mr - gradient;
		vct = vct + lr * ms;
		
		md = vct_to_md(vct);
		
		fh = reshape(peer_units, nhid, 1);
		md.hid_con = fh;
		
		
		
		
		training_data_losses = [training_data_losses, loss(md, datapool.test, wdr)];
		validation_data_losses = [validation_data_losses, loss(md, datapool.validation, wdr)];
		
		if es && validation_data_losses(end) < best_cur.validation_loss,
			best_cur.vct = vct;
			best_cur.validation_loss = validation_data_losses(end);
			best_cur.after_niter = cur_iter;
		end
		
		if mod(cur_iter, round(niter/30)) == 0,
			fprintf('AFTER %d ITERATIONS, TRAINING DATA LOSS IS %f, VALIDATION DATA LOSS IS %f\n', cur_iter, training_data_losses(end), validation_data_losses(end));
		end
		
	end
	
	if es,
		fprintf('VALIDATION LOSS LOWEST AT %d ITERATIONS. EARLY STOPPED', best_cur.after_niter);
		vct = best_cur.vct;
	end
	
	md = vct_to_md(vct);
	%plot
	clf;
	hold on;
	plot(training_data_losses, 'b');
	plot(validation_data_losses, 'r');
	legend('training', 'validation');
	ylabel('loss');
	xlabel('iteration times');
	hold off;
	
	datapool2 = {datapool.test, datapool.validation, datapool.training};
	datanames = {'training', 'validation', 'test'};
	for inx = 1:3,
		dt = datapool2{inx};
		dtname = datanames{inx};
		fprintf('\n LOSS ON DATA %s IS %f\n', dtname, loss(md, dt, wdr));
		if wdr ~= 0,
			fprintf('LOSS WITHOUT WEIGHT DECAY ON %s IS %f\n', dtname, loss(md, dt, 0));
		end
		% classification error rate omitted here.
	end
end

%**********************calculate the gradient*******************
function ret = d_loss_by_d_model(md, dataset, wdr)
	hid_input = md.input_to_hid * dataset.inputs;
	% <number of hidden units> * <number of data cases>
	
	%recurrent part hid_output is the same shape as hid_input
	firstvector = logistic(hid_input(:, 1));% <number of hidden units> * 1
	hid_output = firstvector;
	
	for i = 2 : size(hid_input, 2),
		sp = hid_output(1, end);
		addtail = [hid_output(:,end);sp];
		peerprop = md.hid_con .* addtail(2 : end);
		tempv = logistic(hid_input(:, i) + peerprop);
		hid_output = [hid_output, tempv];
	end
	
	class_input = md.hid_to_class * hid_output;%10*<number of data cases>
	
	%softmax implementation
	class_normalizer = softmax(class_input);
	class_index = class_input - repmat(class_normalizer, [size(class_input, 1), 1]);
	class_prob = exp(class_index);
	
	output_delta = (class_prob - dataset.targets);
	ret.hid_to_class = output_delta * transpose(hid_output);%10*<number of hidden> sum of error*output,witch is derivtive
	ret.hid_to_class = ret.hid_to_class ./ size(dataset.inputs, 2) + wdr * md.hid_to_class;
	
	%backpropagate error in rnn
	back_to_hid = md.hid_to_class' * output_delta;
	%10*nhid' * 10*number of cases, <number of hidden> * <number of cases>, each entry is sum of 10 weights multiply error
	back_first = back_to_hid(:, 1) .* hid_output(:, 1) .* (1 - hid_output(:, 1)); % NOTICE that we dont have the derivtive over weight hid_con here.
	loss_out_hid = back_first; % store d error over d hid_con, <number of hidden> * <number of cases>
	
	if size(back_to_hid) ~= 1,
		ind = randperm(size(back_to_hid) - 1, floor(size(back_to_hid) / 2) ) + 1;
	else
		ind = [2];
	end
	
	for it = ind(1, :),
		tn = hid_output(1, it - 1);
		atail = [hid_output(:, it - 1); tn];
		preopt = atail(2 : end);
		curopt = hid_output(:, it);
		dodw = (1 - curopt) .* curopt .* preopt; % d output over d hid_con
		temps = back_to_hid(:, it) .* dodw;
		loss_out_hid = [loss_out_hid, temps];
	end
	
	ret.hid_con = mean(loss_out_hid, 2);
	
	
	dw3 = back_to_hid .* (1 - hid_output) .* hid_output; % <number of hidden> * <number of cases>
	ret.input_to_hid = dw3 * dataset.inputs'; % <number of hidden> * 256, each is sum of all cases
	ret.input_to_hid = ret.input_to_hid ./ size(dataset.inputs, 2) + wdr * md.input_to_hid;
end

%**************initialize model********************
function ret = createmodel(nhid)
	paranum = (256 + 10 + 1) * nhid;
	ts_vector = cos(0 : (paranum - 1));
	ret = vct_to_md(ts_vector(:) * 0.1);
end

%****************turn a vector into a model******************
function ret = vct_to_md(vct)
	nhid = size(vct, 1) / (256+10+1);
	%note that reshape function will fill the values in first dimention recurrently. p.s. here deem the first number of matrix shape as the number of entries of the first demention
	ret.input_to_hid = transpose(reshape(vct(1 : 256 * nhid), 256, nhid));
	%nhid * 256
	ret.hid_to_class = transpose(reshape(vct(256 * nhid + 1 : 266 * nhid), nhid, 10));
	%10 * nhid
	ret.hid_con = reshape(vct(266 * nhid + 1 : size(vct, 1)), nhid, 1);
	%nhid *1
end

%***************turn a model into a vector*****************
function ret = md_to_vct(md)
	input_to_hid_tran = transpose(md.input_to_hid);
	hid_to_class_tran = transpose(md.hid_to_class);
	hid_con_tran = transpose(md.hid_con);
	ret = [input_to_hid_tran(:); hid_to_class_tran(:); hid_con_tran(:)];
end

%************************loss fuction********************
function ret = loss(md, dataset, wdr)
	hid_input = md.input_to_hid * dataset.inputs;
	% <number of hidden units> * <number of data cases>
	
	%recurrent part hid_output is the same shape as hid_input
	firstvector = logistic(hid_input(:, 1));% <number of hidden units> * 1
	hid_output = firstvector;
	
	for i = 2 : size(hid_input, 2),
		sp = hid_output(1, end);
		addtail = [hid_output(:,end);sp];
		peerprop = md.hid_con .* addtail(2 : end);
		tempv = logistic(hid_input(:, i) + peerprop);
		hid_output = [hid_output, tempv];
	end
	
	class_input = md.hid_to_class * hid_output;%10*<number of data cases>
	
	%softmax implementation
	class_normalizer = softmax(class_input);
	class_index = class_input - repmat(class_normalizer, [size(class_input, 1), 1]);
	class_prob = exp(class_index);
	
	classification_loss = -mean(sum(class_index .* dataset.targets, 1));
	weight_loss = sum(md_to_vct(md) .^2)/2*wdr;
	ret = classification_loss + weight_loss;
end

%*************************logistic*************************
function ret = logistic(hui)
	ret = 1 ./ (1 + exp(-hui));
end

%*************************log(sum(exp(a),1))*****************************
function ret = softmax(jt)
	jtmax = max(jt, [], 1);
	allmax = repmat(jtmax, [size(jt, 1), 1]);
	ret = log(sum(exp(jt - allmax), 1)) + jtmax;
end