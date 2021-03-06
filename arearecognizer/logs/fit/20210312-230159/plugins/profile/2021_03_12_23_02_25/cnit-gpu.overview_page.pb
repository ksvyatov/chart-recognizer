?	`̖???w@`̖???w@!`̖???w@      ??!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`̖???w@1k?SU? v@Au?Hg`@I??Dg?M6@:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"A5.9 % of the total step time sampled is spent on 'Kernel Launch'.*noI?f??1@Qo?I??LW@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	k?SU? v@k?SU? v@!k?SU? v@*      ??!       2	u?Hg`@u?Hg`@!u?Hg`@:	??Dg?M6@??Dg?M6@!??Dg?M6@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?f??1@yo?I??LW@?	"x
Ltraining/Adam/gradients/gradients/conv2d_32/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter????F??!????F??0"x
Ltraining/Adam/gradients/gradients/conv2d_31/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter??(?y??!???2???0"x
Ltraining/Adam/gradients/gradients/conv2d_29/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltersxR??8??!?2u?ٝ?0"x
Ltraining/Adam/gradients/gradients/conv2d_34/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter\IU??#??!?k?$?u??0"x
Ltraining/Adam/gradients/gradients/conv2d_33/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter>??;????!???sv???0"x
Ltraining/Adam/gradients/gradients/conv2d_30/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter?ƹց?!ں?$h??0"x
Ltraining/Adam/gradients/gradients/conv2d_74/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter??S?#?!v??7P&??0"x
Ltraining/Adam/gradients/gradients/conv2d_68/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter??3a??~?!?3?m??0"w
Ktraining/Adam/gradients/gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter:?l?a?~?!I??#??0"w
Ktraining/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterZ8?[p?~?!ϔ??
???0Q      Y@Y?d?*V?K@al?[թqF@"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderateA5.9 % of the total step time sampled is spent on 'Kernel Launch'.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 