## **TensorFlow RDMA源码剖析** 

## **前言**

TensorFlow1.2版本之前，TensorFlow的分布式使用grpc方式来实现，grpc通信方式会导致多次用户态和内核态之间的切换，发送方需要把数据从应用内存拷贝到内核内存，然后传输给接收方，接收方同样需要经过内核内存才能最终获得数据，这就造成了大量的内存带宽和CPU消耗。

而RDMA可以通过网络把数据直接传入计算机的存储区，将数据从一个系统快速移动到远程系统存储器中，而不对操作系统造成任何影响，这样就不需要用到多少计算机的处理功能。它消除了外部存储器复制和文本交换操作，因而能解放内存带宽和CPU周期用于改进应用系统性能。RDMA采用零拷贝网络技术，使NIC 可以直接与应用内存相互传输数据，从而消除了在应用内存与内核内存之间复制数据的需要，内核内存旁路使应用无需执行内核内存调用就可向NIC 发送命令，在不需要任何内核内存参与的条件下， RDMA 请求从用户空间发送到本地NIC，并通过网络发送给远程NIC ，这就减少了在处理网络传输流时内核内存空间与用户空间之间环境切换的次数。

因此，利用RDMA技术可以大大提高计算机系统性能，降低节点间传输数据所需要的延时。在最近的TensorFlow1.2版本中，雅虎公司贡献了RDMA部分的代码，TensorFlow和RDMA的结合可谓是强强联手，使得深度学习分布式训练速度大大加快，提高了数据科学家和算法工程师的工作效率。

## **概述**

TensorFlow的核心是数据流图，计算图的执行就是数据（Tensor）沿着边传递闭包完成流动（Flow）的过程，这也是TensorFlow框架名字的由来。

![pic9](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic9.png)

TensorFlow支持单机和分布式训练，对于单机训练，完成训练的client、master和worker属于同一个进程，client端用户写好的脚本通过session传递给master，master会对任务进行划分并分发给本地的不同设备完成训练，训练的结果会从master返回给client端，最终呈现给用户。对于分布式训练，client、master和worker很可能属于不同的进程，master对图进行优化处理后将子图分发给不同的worker进程完成训练。因此，TensorFlow的分布式训练任务由一组进程（tasks）组成，每个进程都暴露相同的接口，可以调用相同的图执行API，这些tasks中的一部分扮演着参数服务器的角色，被称为ps tasks，其他的tasks用于完成图的训练，被称为worker tasks。多个tasks之间为了协调共同完成图的训练，就要进程之间保持通信（同步/异步），完成参数的更新和下发。

因此，TensorFlow数据流图的执行可以分为三个部分：1、初始化，2、图执行，3、图与图之间数据传输。接下来，将会详细介绍这三个部分中RDMA是怎样工作的。

![pic10](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic10.png)

## **一、源文件列表** 

在正式介绍RDMA工作原理之前，首先列出TensorFlow RDMA模块涉及的源文件，以及依赖的TensorFlow内核源文件，具体列表如下：

RDMA源码位于tensorflow-1.2.0/tensorflow/contrib/verbs 文件夹中

- rdma.h	rdma.cc
- rdma_mgr.h rdma_mgr.cc
- rdma_rendezvous_mgr.h rdma_rendezvous_mgr.cc
- grpc_verbs_client.h grpc_verbs_client.cc
- grpc_verbs_service.h grpc_verbs_service.cc
- grpc_verbs_service_impl.h grpc_verbs_service_impl.cc
- verbs_server_lib.h verbs_server_lib.cc
- verbs_util.h verbs_util.cc
- verbs_service.proto

依赖于tensorflow-1.2.0/tensorflow/core/distributed_runtime 文件夹中

- rpc/grpc_server_lib.h	rpc/grpc_server_lib.cc
- rpc/rpc_rendezvous_mgr.h rpc/rpc_rendezvous_mgr.cc
- base_rendezvous_mgr.h base_rendezvous_mgr.cc
- rendezvous_mgr_interface.h
- server_lib.h server_lib.cc

依赖于tensorflow-1.2.0/tensorflow/core/framework 文件夹中

- op_kernel.h op_kernel.cc
- rendezvous.h rendezvous.cc

依赖于tensorflow-1.2.0/tensorflow/core/kernel 文件夹中

- sendrecv_ops.h sendrecv_ops.cc

## **二、RDMA 初始化** 

故事还要从TensorFlow的分布式讲起，下面一段代码是TensorFlow官网给出的Distributed TensorFlow教程。

```cpp
import argparse
import sys
import tensorflow as tf
FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
      # Build model...
      loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()
      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
```

这段代码和单机版的代码一样，都是先构建一个graph，然后定义损失和梯度等，最后在session中完成训练。唯一的不同是，定义了

```cpp
server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
```

tf.train.Server 封装了一组设备和tf.session，使用Server可以实现分布式训练，同一个 cluster中的Server可以进行通信。这行代码其实默认了一个参数，更为完整的应该是

```cpp
server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index,
                           protocol='grpc')
```

这里多了一个叫做protocol的参数，这个参数指定了server采用grpc协议进行通信。而如果我们想使用RDMA进行通信的话需要指定

```cpp
server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index,
                           protocol='grpc+verbs')
```

在Python端，tf.train.Server只是定义了计算图中的一个算子，当图通过session提交给后端master后，graph开始执行，server会进行初始化操作，server设置阶段，会调用GrpcServer类完成初始化、开启服务等操作，GrpcServer是ServerInterface的派生类，ServerInterface是一个接口类，声明了start（），stop（），join（）等接口，start用于开启service（服务），stop用于终止服务，join会阻塞直到server结束。而GrpcServer将这些接口具体实现，同时实现init（），这个init会根据客户端定义的cluster列表完成server初始化，在这之后server就有了cluster中个节点的IP、port信息，可以进行相应的grpc通信了。

![img](http://www.literacystudies.com/src2/tensorflow/core/classtensorflow_1_1_server_interface.png)

GrpcServer的状态转换图如下

![pic5](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic5.png)

```cpp
tf.train.ClusterSpec({    (cluster列表)
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
```

## **(1) verbs_server_lib.h**

到这里，我们了解了TensorFlow基于传统的gRPC是如何工作的。接下来我们介绍TensorFlow基于RDMA的初始化工作原理，对于RDMA，server初始化时会调用verbs_server_lib.h中的class VerbsServer，VerbsServer是GrpcServer的派生类，同样是实现ServerInterface中的接口函数以及init函数。

- VerbsServer数据成员：

```cpp
  RdmaMgr* rdma_mgr_;
  // Guards state transitions.
  mutex mu_;
  GrpcVerbsService* verbs_service_ = nullptr;
  std::unique_ptr<Thread> verbs_thread_ GUARDED_BY(mu_);
  GrpcChannelCache* channel_cache_ = nullptr;
```

- VerbsServer函数成员：

```cpp
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);
  // Implementations of ServerInterface methods.
  Status Start() override;
  Status Join() override;
 protected:
  Status Init(ServiceInitFunction service_func,
              RendezvousMgrCreationFunction rendezvous_mgr_func);
  Status ChannelCacheFactory(const ServerDef& server_def,
                             GrpcChannelCache** channel_cache);
```

Init()中调用GrpcServer::Init完成grpc协议的初始化，这样server有了集群中各个节点的地址信息，可以进行grpc调用，然后创建了一个RdmaMgr（RdmaMgr用于管理RDMA的channels和adapter，关于RdmaMgr详情可参考“RDMA 数据通信”章节），并为verbs_service和rdma_rendezvous_mgr设置RdmaMgr。

```cpp
Status VerbsServer::Init(ServiceInitFunction service_func,
                         RendezvousMgrCreationFunction rendezvous_mgr_func) {
  Status s = GrpcServer::Init(service_func, rendezvous_mgr_func);
  {
    mutex_lock l(mu_);
    CHECK_EQ(verbs_state_, DISCONNECTED);
    CHECK(ChannelCacheFactory(server_def(), &channel_cache_).ok());
    rdma_mgr_ = new RdmaMgr(worker_env(), channel_cache_);
    // set rdma_mgr for verbs_service and rdma_rendezvous_mgr
    verbs_service_->SetRdmaMgr(rdma_mgr_);
    dynamic_cast<RdmaRendezvousMgr*>(worker_env()->rendezvous_mgr)
        ->SetRdmaMgr(rdma_mgr_);
  }
  return s;
}
```

Start()中调用GrpcServer::Start()开启grpc的服务，然后初始化verbs_thread_线程，使其开启verb service 服务，最后调用rdma_mgr的SetupChannels()来完成RDMA channel的初始化。相信此时您已经有点晕了，没关系，听我慢慢道来。首先，这段代码还是倒过来看比较好，SetupChannels是干什么的呢？RDMA要想工作得做一些准备工作吧！RDAM 使用RDMA channel来完成端到端的通信，而通信得需要地址吧，最最开始的时候，RDMA channel根本不知道其它节点的地址啊，这个SetupChannels就是完成RDMA channel的初始化，它通过gRPC来获得远程端的地址，而想进行GRPC调用就需要server端开启相应的服务（返回地址给client），而verbs_thread做的就是开启一个线程实现“返回地址信息“的服务，HandleRPCsLoop就是服务的具体实现。至此start就介绍完了，这个非常关键，**示意图和流程图见下图** 。

```cpp
Status VerbsServer::Start() {
  Status s = GrpcServer::Start();
  {
    mutex_lock l(mu_);
    if (verbs_state_ == DISCONNECTED) {
      // verbs_thread needs to be initiated
      // before rdma_mgr sets up the rdma channels.
      verbs_thread_.reset(worker_env()->env->StartThread(
          ThreadOptions(), "TF_verbs_service",
          [this] { verbs_service_->HandleRPCsLoop(); }));
      rdma_mgr_->SetupChannels();
      verbs_state_ = CONNECTED;
    }
  }
  return s;
}
```

![img](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic4.png)

## **(2) grpc_verbs_service.h**

前面说道RdmaMgr通过调用gRPC服务来初始化RDMA channel，使RDMA channel拥有远程节点的地址信息，而grpc_verbs_service.h中的class GrpcVerbsService就定义了这个服务的具体实现。

- GrpcVerbsService的数据成员

```cpp
  ::grpc::ServerCompletionQueue* cq_;
  grpc::VerbsService::AsyncService verbs_service_;
  mutex shutdown_mu_;
  bool is_shutdown_ GUARDED_BY(shutdown_mu_);
  ::grpc::Alarm* shutdown_alarm_;
  // not owned
  RdmaMgr* rdma_mgr_;
  const WorkerEnv* const worker_env_;
```

- GrpcVerbsService的函数成员

```cpp
  void HandleRPCsLoop() override;
  void Shutdown() override;
  void SetRdmaMgr(RdmaMgr* rdma_mgr) { rdma_mgr_ = rdma_mgr; }
 private:
  template <class RequestMessage, class ResponseMessage>
  using WorkerCall = Call<GrpcVerbsService, grpc::VerbsService::AsyncService,
                          RequestMessage, ResponseMessage>;
  void GetRemoteAddressHandler(
      WorkerCall<GetRemoteAddressRequest, GetRemoteAddressResponse>* call);
  Status GetRemoteAddressSync(const GetRemoteAddressRequest* request,
                              GetRemoteAddressResponse* response);
```

上节提到VerbsServer::Start调用HandleRPCsLoop来开启服务

```cpp
void GrpcVerbsService::HandleRPCsLoop() {
  for (int i = 0; i < 10; ++i) {
    ENQUEUE_REQUEST(GetRemoteAddress, false);
  }
  void* tag;
  bool ok;
  while (cq_->Next(&tag, &ok)) {
    UntypedCall<GrpcVerbsService>::Tag* callback_tag =
        static_cast<UntypedCall<GrpcVerbsService>::Tag*>(tag);
    if (callback_tag) {
      callback_tag->OnCompleted(this, ok);
    } else {
      cq_->Shutdown();
    }
  }
}
```

HandleRPCsLoop调用宏ENQUEUE_REQUEST，宏ENQUEUE_REQUEST调用GetRemoteAddressHandler，GetRemoteAddressHandler调用GetRemoteAddressSync，GetRemoteAddressSync是服务的具体实现，client希望获得server的地址信息，GetRemoteAddressSync把server的地址以及RDMA channel中各buffer的地址、key信息通过GetRemoteAddressResponse发送给client，至此client获得了远程节点的地址信息，可以进行channel初始化以及愉快的进行RDMA数据传输了。

```cpp
Status GrpcVerbsService::GetRemoteAddressSync(
    const GetRemoteAddressRequest* request,
    GetRemoteAddressResponse* response) {
  // analyzing request
  // the channel setting part is redundant.
  const string remote_host_name = request->host_name();
  RdmaChannel* rc = rdma_mgr_->FindChannel(remote_host_name);
  CHECK(rc);
  RdmaAddress ra;
  ra.lid = request->channel().lid();
  ...
  rc->SetRemoteAddress(ra, false);
  rc->Connect();
  int i = 0;
  int idx[] = {1, 0, 3, 2};
  std::vector<RdmaBuffer*> mb(rc->message_buffers());
  CHECK_EQ(request->mr_size(), 4);
  for (const auto& mr : request->mr()) {
    // hence idx[] = {1, 0, 3, 2}.
    RdmaBuffer* rb = mb[idx[i]];
    RemoteMR rmr;
    rmr.remote_addr = mr.remote_addr();
    rmr.rkey = mr.rkey();
    rb->SetRemoteMR(rmr, false);
    i++;
  }
  CHECK(i == RdmaChannel::kNumMessageBuffers);
  // setting up response
  response->set_host_name(
      worker_env_->session_mgr->LegacySession()->worker_name);
  Channel* channel_info = response->mutable_channel();
  channel_info->set_lid(rc->self().lid);
  ...
  for (int i = 0; i < RdmaChannel::kNumMessageBuffers; i++) {
    MemoryRegion* mr = response->add_mr();
    mr->set_remote_addr(reinterpret_cast<uint64>(mb[i]->buffer()));
    mr->set_rkey(mb[i]->self()->rkey);
  }
  return Status::OK();
}
```

## **(3) grpc_verbs_service_impl.h**

client想要获得远程节点的地址，就需要调用gRPC，远程节点的server会通过相应的服务将地址信息返回给client。但是，这个gRPC协议是怎样定义的呢，如何实现client的gRPC调用呢。这个就是grpc_verbs_service_impl.h中class VerbsService GRPC_FINAL 完成的任务啦。VerbsService基于//tensorflow/contrib/verbs/verbs_service.proto文件实现了tensorflow.VerbsService，

```cpp
class VerbsService GRPC_FINAL {   // GPRC同步调用
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status GetRemoteAddress(
        ::grpc::ClientContext* context, const GetRemoteAddressRequest& request,
        GetRemoteAddressResponse* response) = 0;
  };
  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status GetRemoteAddress(
        ::grpc::ClientContext* context, const GetRemoteAddressRequest& request,
        GetRemoteAddressResponse* response) GRPC_OVERRIDE;
   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    const ::grpc::RpcMethod rpcmethod_GetRemoteAddress_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr< ::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());
  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    void RequestGetRemoteAddress(
        ::grpc::ServerContext* context, GetRemoteAddressRequest* request,
        ::grpc::ServerAsyncResponseWriter<GetRemoteAddressResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
};
```

整个实现与通常的GPRC调用没什么两样，如果您不熟悉GPRC调用可以参考下面这个链接

[GPRC同步调用分析](https://www.sailsxu.com/?p=608#2)

## **三、图执行**

接下来，我们介绍单机环境下计算图是如何执行的。计算图中各节点的执行顺序遵循于节点间的依赖，对于一个节点，如果它所有依赖的节点都已经计算完成，那么它就可以开始计算。以下图为例，节点D依赖于节点C和常数节点1，因此节点D不能执行，而节点C又依赖于节点A和节点B，因此节点C暂时也不能执行，而节点A和节点B没有依赖，节点A和B会被加入到ready队列中等待执行，当节点A和B完成计算后，节点C的依赖数从2减为0，然后节点C被加入到ready队列中等待执行，以此类推。

![pic11](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic11.png)

对于跨设备和分布式环境，TensorFlow会在不同设备的两个相邻节点之间添加Send和Recv节点，通过Send和Recv之间进行通信来达到op之间通信的效果，如下图所示。图中还涉及到一个优化问题，即a->b和a->c需要建立两组send/recv连接的，但两组连接是可以共用的，所以合并成一组连接。接下来，我们介绍Send和Recv节点的具体实现。

![pic12](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic12.png)

这一节的主人公就是SendOp和RecvOp这两个算子（节点）。SendOp和RecvOp定义在sendrecv_ops.h中，与其它几百个算子一样，它们的实现都位于tensorflow/core/kernels/文件夹下。与其它几百个算子一样，SendOp和RecvOp的具体实现就是Compute（）函数，它定义了send和recv算子在计算图中的行为。SendOp和RecvOp在分布式训练中至关重要，因为它们定制了单机多卡/多机多卡的计算图之间如何完成tensor通信。
![pic8](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic1.png)

### **（1）SendOp**

```cpp
class SendOp : public OpKernel {
 public:
  explicit SendOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;
 private:
  string key_prefix_;
  TF_DISALLOW_COPY_AND_ASSIGN(SendOp);
};

class RecvOp : public AsyncOpKernel {
 public:
  explicit RecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
 private:
  string key_prefix_;
  TF_DISALLOW_COPY_AND_ASSIGN(RecvOp);
};
```

SendOp::Compute是Send算子的具体实现，它的核心就是调用rendezvous()的Send函数发送tensor，这个Send函数是虚函数，它调用的是Rendezvous指针指向的实例，然而，不管Rendezvous指针指向谁，最终都是调用LocalRendezvousImpl的send（），也就是说，不管是单机还是分布式（gRPC or RDMA），Send函数都是一样的，就是把tensor放在本地，被动的等待消费者（RecvOp）发出请求。详细的调用栈见下图。

```cpp
void SendOp::Compute(OpKernelContext* ctx) {
  OP_REQUIRES(
      ctx, ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."));
  Rendezvous::ParsedKey parsed;
  GetRendezvousKey(key_prefix_, ctx->frame_iter(), &parsed.buf_);
  VLOG(2) << "Send " << parsed.buf_;
  OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed.buf_, &parsed));
  // The device context may be passed between the Send/Recv
  // boundary, so that the device context used to produce the Tensor
  // is used when performing the copy on the recv side (which may be
  // a different device).
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->input_alloc_attr(0);
  OP_REQUIRES_OK(ctx, ctx->rendezvous()->Send(parsed, args, ctx->input(0),
                                              ctx->is_input_dead()));
}
```

![pic8](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic8.png)

### **（2）RecvOp**

RecvOp::ComputeAsync是Recv算子的具体实现，它的核心就是调用rendezvous()->RecvAsync向生产者（SendOp）请求tensor，RecvAsync是一个虚函数，也就是说，对于不同的派生类对象（本地、gPRC、RDMA）RecvAsync的实现是不同的，详细的调用栈建下图。

```cpp
void RecvOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  OP_REQUIRES(
      ctx, ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."));
  Rendezvous::ParsedKey parsed;
  GetRendezvousKey(key_prefix_, ctx->frame_iter(), &parsed.buf_);
  ......
  Rendezvous::DoneCallback done_cb = std::bind(
      [ctx](DoneCallback done,
            // Begin unbound arguments.
            const Status& s, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& val,
            bool is_dead) {
        ctx->SetStatus(s);
        if (s.ok()) {
          // 'ctx' allocates the output tensor of the expected type.
          // The runtime checks whether the tensor received here is
          // the same type.
          if (!is_dead) {
            ctx->set_output(0, val);
          }
          *ctx->is_output_dead() = is_dead;
        }
        done();
      },
      std::move(done), _1, _2, _3, _4, _5);
  ctx->rendezvous()->RecvAsync(parsed, args, std::move(done_cb));
}
```

![pic1](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic7.png)

RecvOp最终调的是BaseRemoteRendezvous中实现的RecvAsync，而RecvAsync会根据tensor传输的src和dst位置决定调用哪种实现，当src和dst位于一个节点时，RecvAsync会调用LocalRendezvousImpl的RecvAsync方法完成本地Recv操作。当src和dst位于不同节点，RecvAsync会调用RecvFromRemoteAsync完成远程Recv操作，RecvFromRemoteAsync是一个接口函数，它的具体实现取决于用户选择使用哪种通信方式（gRPC / RDMA），当用gRPC通信时，调用RpcRemoteRendezvous中的RecvFromRemoteAsync实现。当用RDMA通信时，调用RdmaRemoteRendezvous中的RecvFromRemoteAsync实现。RdmaRemoteRendezvous::RecvFromRemoteAsync就是我们下一节中着重要介绍的使用RDMA完成recv算子向send算子请求tensor的具体实现。

```cpp
void BaseRemoteRendezvous::RecvAsync(const ParsedKey& parsed,
                                     const Rendezvous::Args& recv_args,
                                     DoneCallback done) {
  ......
  // Are src and dst in the same worker?
  if (IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_->RecvAsync(
        parsed, recv_args,
        [this, parsed, done](
            const Status& status, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& in, bool is_dead) {
          ......
        });
    return;
  } else {
    RecvFromRemoteAsync(parsed, recv_args, std::move(done));
  }
}
```

至此，TensorFlow计算流图中的Send节点和Recv节点是如何完成通信的我们就剖析完了。总的来说就是，对于单机和分布式，Send节点的操作大体相同，都是把计算好的tensor放在本地，被动的等待Recv节点（本地/远程）发出请求。而Recv节点会根据需要通信的两个节点是否处于同一个物理节点以及通信模式采用gRPC还是效率更高的RDMA来决定最终调用那个实现。如果用户选择了RDMA方式通信，那么调用的就是接下来要介绍的通信方法了。

## **四、RDMA 数据通信**

在“RDMA 初始化”一节中我们剖析了TensorFlow RDMA的初始化工作，RDMA使用gRPC调用来获得远程端的地址以及相应buffer的内存地址和key，在这之后，RDMA channel就拥有了集群各节点的地址信息，可以进行消息通信和数据（tensor）传输了。那么TensorFlow如何使用RDMA进行通信呢，这就是这部分要介绍的内容了。

我们知道了TensorFlow的执行其实就是数据流图的执行，单机的图执行没什么好说的，就是根据节点间的依赖关系来执行，对于分布式来说，图和图之间是如何通信的呢？ TensorFlow会在需要通信的节点路径上添加send和recv算子，send算子用于发送tensor，而recv算子用于接收tensor，这两个算子封装了通信的实现。如下图所示，send 操作是被动的, 它仅仅将一个tensor放入到local out-going 表中。 receive 操作 才是真正的启动 tensor 的传输。

![img](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic1.png)

当一个tensor准备进行传输，它先会被转化为TensorProto格式，然后这个proto被序列化为byte数组，拷贝到绑定的buffer中。buffer中的内容将会通过RDMA write传输到远程节点。对于远程节点，处理的流程正好相反。尽管原始的tensor是存放在device上的，TensorProto是存在在host memory上的，所有的绑定buffer都在host memory中被分配。

TensorFlow RDMA的基本组件包括 **RDMA adapter**、**RDMA channel**、**RDMA buffer**、**RDMA manager**、**RDMA rendezvous manager**、**RDMA rendezvous**等，下面概要介绍一下这几个组件。

- **RDMA adapter:** 用于 RDMA 通信. 它包含多个 channels 和 buffers. adapter负责处理不同的RDMA messages。
- **RDMA channel:** Responsible for RDMA connection to a particular node. channel管理多个 buffers. 一个 channel 有一个 callback table ，这个表用于存放所有的请求tensor的callback函数。
- **RDMA buffer:** 负责发送和接受数据。它有固定大小的memory来存储数据。它有一个队列来存放排队的jobs。有三种类型的buffers, 分别是message buffer, ACK buffer 和 tensor buffer。一个 channel 有两个message buffers, 两个ack buffers 和很多 tensor buffers.
- **RDMA manager:** 管理 adapter 和 channels, 包括 channel 创建, channel 设置（通过GRPC服务）, channel 查询, 等.
- **RDMA rendezvous manager:** 管理多个 rdma rendezvous。
- **RDMA rendezvous:** BaseRemoteRendezvous 的派生类。 这个类是 "send" 和 "recv" 算子的背后实现。当 sendrecv_op 算子想要 send 或 receive 一个tensor, 它会分别调用 rendezvous的 "send" and "recv" 函数. Rendezvous 通过"step_id"来识别, step_id是一个随机数, 因此不同迭代的tensor不会被混淆。

## **（1）rdma.h rdma.cc**

- ### **class RdmaAdapter**

  - 数据成员：

    ```cpp
      ibv_context* context_;
      // ibverbs protection domain
      ibv_pd* pd_;
      // Completion event channel, to wait for work completions
      ibv_comp_channel* event_channel_;
      // Completion queue, to poll on work completions
      ibv_cq* cq_;
      // Pre-allocated work completions array used for polling
      ibv_wc wc_[MAX_CONCURRENT_WRITES * 2];
      // worker env for thread
      const WorkerEnv* worker_env_;
      // thread for cq.
      std::unique_ptr<Thread> polling_thread_;
    ```

  - 函数成员：

    ```cpp
    void Process_CQ();
    ```

  RdmaAdapter负责处理不同的RDMA messages请求，它会使用ibv_comp_channel监听完成的事件，将完成的事件加入到ibv_cq完成队列之中，然后轮询处理完成的事件。其中ibv_comp_channel和ibv_cq都是很重要的RDMA API，关于RDMA API可以参考这个链接

  RDMA Aware Networks Programming.pdf

  。 Process_CQ()就定义了RDMA如何处理这些事件

  ```cpp
  // Function to process incoming messages
  void RdmaAdapter::Process_CQ() {
    while (true) {
      ibv_cq* cq;
      void* cq_context;
      CHECK(!ibv_get_cq_event(event_channel_, &cq, &cq_context));
      CHECK(cq == cq_);
      ibv_ack_cq_events(cq, 1);
      CHECK(!ibv_req_notify_cq(cq_, 0));
      int ne =
          ibv_poll_cq(cq_, MAX_CONCURRENT_WRITES * 2, static_cast<ibv_wc*>(wc_));   
      CHECK_GE(ne, 0);
      for (int i = 0; i < ne; ++i) {
        if (wc_[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          RdmaChannel* rc = reinterpret_cast<RdmaChannel*>(wc_[i].wr_id); 
          // put back a recv wr.
          rc->Recv();
          // imm_data is the index of RX buffer in the buffer table.
          uint32_t imm_data = wc_[i].imm_data;
          RdmaBuffer* rb = rc->FindBuffer(imm_data);
          RdmaMessage rm;
          RdmaMessage::ParseMessage(rm, rb->buffer_);
          VLOG(2) << "recv RDMA message: " << MessageTypeToString(rm.type_);
          if (rm.type_ == RDMA_MESSAGE_ACK) {
            .....
          } else if (rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) {
            .....
          } else if (rm.type_ == RDMA_MESSAGE_BUFFER_IDLE) {
            .....
          } else if (rm.type_ == RDMA_MESSAGE_BUFFER_REQUEST) {
            .....
          } else if (rm.type_ == RDMA_MESSAGE_BUFFER_RESPONSE) {
            .....
          } else if (rm.type_ == RDMA_MESSAGE_TENSOR_WRITE) {
            .....
          }
        } else if (wc_[i].opcode == IBV_WC_RDMA_WRITE) {
  		  .....
        }
      }
    }
  }
  ```

  具体的处理流程描述如下图：

  ![img](http://git.code.oa.com/paichen/TensorFlow-source/raw/master/TensorFlow%20RDMA/pic/pic2.png)

- ### **class RdmaChannel**

  - 数据成员：

    ```cpp
      const RdmaAdapter* adapter_;
      RdmaAddress self_;
      string local_name_;
      string remote_name_;
      ibv_qp* qp_;
      mutex mu_;
      bool connected_ GUARDED_BY(bt_mu_) = false;
      RdmaAddress remote_ GUARDED_BY(bt_mu_);
      bool remote_set_ GUARDED_BY(bt_mu_) = false;
      mutex ct_mu_;
      typedef std::unordered_map<string, std::function<void()> > CallbackTable;
      CallbackTable callback_table_ GUARDED_BY(ct_mu_);
      mutex bt_mu_;
      typedef std::unordered_map<unsigned int, RdmaBuffer*> BufferTable;
      BufferTable buffer_table_ GUARDED_BY(bt_mu_);
      typedef std::unordered_map<uint32_t, string> BufferIndexNameTable;
      BufferIndexNameTable buffer_index_name_table_ GUARDED_BY(bt_mu_);
      typedef std::unordered_map<string, uint32_t> BufferNameIndexTable;
      BufferNameIndexTable buffer_name_index_table_ GUARDED_BY(bt_mu_);
      RdmaBuffer* tx_message_buffer_;
      RdmaBuffer* rx_message_buffer_;
      RdmaBuffer* tx_ack_buffer_;
      RdmaBuffer* rx_ack_buffer_;
      std::vector<RdmaBuffer*> message_buffers_;
    ```

  - 函数成员：

    ```cpp
      void Connect(const RdmaAddress& remoteAddr);
      void Connect();
      void Recv();
      RdmaBuffer* FindBuffer(const uint32_t index);
      RdmaBuffer* FindBuffer(const string& name);
      RdmaBuffer* FindOrCreateBuffer(const string& name,
                                     BufferType buffer_type = TENSOR);
      uint32_t LookupBufferIndex(const string& buffer_name);
      void SetRemoteAddress(const RdmaAddress& ra, bool override);
      void InsertRecvCallback(const string& key, std::function<void()> recv_done);
      void RemoveRecvCallback(const string& key);
      void RunRecvCallback(const string& key);
    ```

    RdmaChannel负责端到端的通信，RdmaChannel拥有ibv_qp（RDMA API）来实现RDMA通信，拥有远程端的地址，还拥有两个ack buffer、两个message buffer、多个tensor buffer。Connect（）用于建立到远程端的连接，Recv（）调用ibv_post_recv来实现RDMA接收消息，FindBuffer（）用于根据索引或名字找到相应的buffer，FindOrCreateBuffer（）用于根据参数type创建相应的buffer（type可以使ack、message、tensor），InsertRecvCallback（）会将callback函数放入callback表中，当Recv端接收到tensor后，会调用这个callback函数（RunRecvCallback）。

    ```cpp
    // Adding tokens to the completion queue
    // Tokens are needed to process future messages.
    void RdmaChannel::Recv() {
      struct ibv_recv_wr wr;
      memset(&wr, 0, sizeof(wr));
      wr.wr_id = (uint64_t)this;
      struct ibv_recv_wr* bad_wr;
      CHECK(!ibv_post_recv(qp_, &wr, &bad_wr)) << "Failed to post recv";
    }
    ```

- ### **class RdmaBuffer**

  - 数据成员：

  ```cpp
    const RdmaChannel* channel_;
    void* buffer_ = nullptr;
    bool buffer_on_host_ = true;
    size_t size_ = 0;
    const string name_;
    ibv_mr* self_ = nullptr;
    mutex mu_;
    RemoteMR remote_;
    std::queue<string> queue_ GUARDED_BY(mu_);
    BufferStatus local_status_ GUARDED_BY(mu_) = none;
    BufferStatus remote_status_ GUARDED_BY(mu_) = none;
  ```

  - 函数成员：

  ```cpp
    void FreeBuffer();
    void EnqueueItem(string Item);
    virtual void SendNextItem(){};
    void CreateCPUBuffer(size_t size, bool lock = true);
    void SetRemoteMR(RemoteMR rmi, bool override);
    uint32_t LookupBufferIndex(const string& buffer_name) ;
    void Write(uint32_t imm_data, size_t buffer_size);
  ```

  RdmaBuffer表示用于RDMA读和写的buffer，RdmaBuffer有三个派生类，分别是RdmaAckBuffer、RdmaMessageBuffer和RdmaTensorBuffer，RdmaMessageBuffer负责发送 message ，比如请求一个tensor等等。一旦一个message被发送，message的接收方需要通过RdmaAckBuffer发送一个ack来释放发送方的message buffer。一个RdmaAckBuffer和唯一的RdmaMessageBuffer绑定。RdmaTensorBuffer负责发送tensor，tensor的接收方需要返回一个message来释放发送方的buffer。

  Write函数通过调用RDMA api：ibv_post_send完成Rdma-Write，将buffer中的内容通过RDMA写入远程节点

  ```cpp
  void RdmaBuffer::Write(uint32_t imm_data, size_t buffer_size) {
    struct ibv_sge list;
    list.addr = (uint64_t)buffer_;
    list.length = buffer_size;
    list.lkey = self_->lkey;
    //An SR(send request) defines how much data will be sent, from where, how and, with RDMA, to where.
    //struct ibv_send_wr is used to implement SRs.
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)this;
    wr.sg_list = &list;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = imm_data;
    wr.wr.rdma.remote_addr = (uint64_t)remote_.remote_addr;
    wr.wr.rdma.rkey = remote_.rkey;
    struct ibv_send_wr* bad_wr;
    CHECK(!ibv_post_send(channel_->qp_, &wr, &bad_wr)) << "Failed to post send";
  }
  ```

  CreateCPUBuffer函数在CPU内存开辟空间来存放相应类型的buffer，目前版本只支持buffer on CPU，不支持RDMABuffer on device，处于TODO阶段。

  三种buffer的区别在于SendNextItem函数的实现上，RdmaAckBuffe的SendNextItem会创建一个name为rx_ack_buffer，type为RDMA_MESSAGE_ACK的message，然后通过write函数完成RDMA写。RdmaMessageBuffer的SendNextItem会将本地节点和远程节点状态设置为繁忙，然后从buffer的事件队列中取出队列头message，通过write函数完成RDMA写。

- ### struct RdmaMessage

RDMA message格式

| type | name_size | name | step_id | buffer_size | remote_addr | rkey | is_dead | data_type | tensor_shape | tensor_bytes | tensor_buffer |
| ---- | --------- | ---- | ------- | ----------- | ----------- | ---- | ------- | --------- | ------------ | ------------ | ------------- |
|      |           |      |         |             |             |      |         |           |              |              |               |

RDMA messages 的6种类型

- RDMA_MESSAGE_ACK
- RDMA_MESSAGE_BUFFER_IDLE
- RDMA_MESSAGE_BUFFER_REQUEST
- RDMA_MESSAGE_BUFFER_RESPONSE
- RDMA_MESSAGE_TENSOR_REQUEST
- RDMA_MESSAGE_TENSOR_WRITE

## **（2）rdma_mgr.h rdma_mgr.cc**

- ### **class RdmaMgr**

  - 数据成员

    ```cpp
      string local_worker_;
      size_t num_remote_workers_;
      const WorkerEnv* const worker_env_;
      GrpcChannelCache* const channel_cache_;
      RdmaAdapter* rdma_adapter_;
      typedef std::unordered_map<string, RdmaChannel*> ChannelTable;
      ChannelTable channel_table_;
    ```

    ​

  - 函数成员

    ```cpp
      RdmaChannel* FindChannel(const string& key);
      void SetupChannels();
      const string& local_worker();
    ```

    RdmaMgr负责管理RdmaAdapter和众多RdmaChannel，RdmaMgr包含local_worker_（本机）名和远程节点的数量，ChannelTable用于存放所有与远程节点关联的RdmaChannel。FindChannel函数用于根据给定的名（key）在ChannelTable中查找相应的RdmaChannel，SetupChannels函数非常重要，这个函数会调用grpc_verbs_client.h中的GrpcVerbsClient类来实现RDMA的初始化，这也是“RDMA初始化”小节的重点。只有RdmaMgr完成了SetupChannels，本地节点上的RdmaChannels才拥有了远程节点的地址以及虚拟内存地址、key信息，才可以进行之后的消息、数据传输工作。

    ```cpp
    void RdmaMgr::SetupChannels() {
      for (const auto& p : channel_table_) {
        string worker_name = p.first;
        RdmaChannel* rc = p.second;
        GetRemoteAddressRequest req;
        GetRemoteAddressResponse resp;
        SharedGrpcChannelPtr client_channel =
            channel_cache_->FindWorkerChannel(worker_name);
        GrpcVerbsClient* client = new GrpcVerbsClient(client_channel);    
        ......
        for (int i = 0; i < RdmaChannel::kNumMessageBuffers; i++) {
          MemoryRegion* mr = req.add_mr();
          mr->set_remote_addr(
              reinterpret_cast<uint64_t>(rc->message_buffers_[i]->buffer_));
          mr->set_rkey(rc->message_buffers_[i]->self_->rkey);
        }
        Status s = client->GetRemoteAddress(&req, &resp);
        if (s.ok()) {
          CHECK(worker_name.compare(resp.host_name()) == 0);
          RdmaAddress ra;
          ra.lid = resp.channel().lid();
          ......
          rc->SetRemoteAddress(ra, false);
          rc->Connect();
          int i = 0;
          int idx[] = {1, 0, 3, 2};
          for (const auto& mr : resp.mr()) {
            RdmaBuffer* rb = rc->message_buffers_[idx[i]];
            RemoteMR rmr;
            rmr.remote_addr = mr.remote_addr();
            rmr.rkey = mr.rkey();
            rb->SetRemoteMR(rmr, false);
            i++;
          }
          CHECK(i == RdmaChannel::kNumMessageBuffers);
        } 
        delete client;
      }
    }
    ```

## **（3）rdma_rendezvous_mgr.h rdma_rendezvous_mgr.cc**

- ### **class RdmaRendezvousMgr**

  负责管理一组local rendezvous实例class RdmaRemoteRendezvous**

- ### **class RdmaRemoteRendezvous**

  - 数据成员

    ```cpp
    RdmaMgr* rdma_mgr_;
    ```

  - 函数成员

    ```cpp
    void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                               const Rendezvous::Args& args,
                               DoneCallback done) override;
    ```

    RdmaRemoteRendezvous是BaseRemoteRendezvous派生类，是recv算子的背后实现，RecvFromRemoteAsync函数定义了recv算子向send算子请求tensor的具体实现。首先验证请求的源和远程设备有效，然后根据远程设备名调用RdmaMgr的FindChannel获得相应的RdmaChannel，通过InsertRecvCallback为RdmaChannel设置相应的recv事件回调函数，回调函数主要就是接收tensor然后回复send端，告知send端已经收到tensor可以释放相应buffer了。在设置了回调函数后，就可以向send端发出tensor请求了——SendNextItem。具体的流程可以参照上图。

    ```cpp
    void RdmaRemoteRendezvous::RecvFromRemoteAsync(
        const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
        DoneCallback done) {
      Status s;
      // parse src_name and dst_name
      string src_name, dst_name, unused;
      if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_name,
                                            &unused)) {
        s = errors::Internal("Could not parse src name.");
      }
      CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
      ......
      if (!s.ok()) {
        done(s, Args(), recv_args, Tensor{}, false);
        return;
      }
      CHECK(dst_name.compare(rdma_mgr_->local_worker()) == 0);
      RdmaChannel* rc = rdma_mgr_->FindChannel(src_name);
      string key(std::move(parsed.FullKey().ToString()));
      string key_with_step_id = VerbsUtil::AppendStepidToKey(key, step_id_);
      // insert callback
      rc->InsertRecvCallback(key_with_step_id, [this, key, key_with_step_id, rc,
                                                recv_args, parsed, done]() {
        Status s;
        Device* src_dev;
        s = env_->device_mgr->LookupDevice("CPU:0", &src_dev);
        CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
        if (!s.ok()) {
          done(s, Args(), recv_args, Tensor(), true);
          return;
        }
        ......
        RdmaBuffer* rb = rc->FindBuffer(key);
        RdmaMessage rm;
        CHECK(rb->size_ >= RdmaMessage::kMessageTotalBytes);
        RdmaMessage::ParseMessage(rm, rb->buffer_);
        CHECK(rm.type_ == RDMA_MESSAGE_TENSOR_WRITE);
        Tensor val;
        if (!rm.is_dead_) {
          void* input = static_cast<char*>(rb->buffer_) +
                        RdmaMessage::kTensorBufferStartIndex;
          TensorProto proto;
          CHECK(rm.tensor_bytes_ + RdmaMessage::kTensorBufferStartIndex <=
                rb->size_);
          CHECK(ParseProtoUnlimited(&proto, input, rm.tensor_bytes_))
              << "fail to parse proto from array";
          s = dst_dev->MakeTensorFromProto(proto, recv_args.alloc_attrs, &val);
        }
        rc->RemoveRecvCallback(key_with_step_id);
        // create message
        RdmaMessage br;
        br.type_ = RDMA_MESSAGE_BUFFER_IDLE;
        br.name_size_ = key.size();
        br.name_ = key;
        string message = RdmaMessage::CreateMessage(br);
        RdmaBuffer* tb = rc->tx_message_buffer_;
        tb->EnqueueItem(message);
        tb->SendNextItem();
        done(s, Args(), recv_args, val, rm.is_dead_);
      });
      // append key to message queue
      RdmaBuffer* rb = rc->tx_message_buffer_;
      RdmaMessage rm;
      ... ...
      string message = RdmaMessage::CreateMessage(rm);
      rb->EnqueueItem(message);
      rb->SendNextItem();
    }
    ```

    ## **总结**

    利用RDMA可以大大提高网络通信的效率，TensorFlow和RDMA的结合可谓是强强联手，在开源RDMA这波大潮影响下，各家深度学习、机器学习框架推出RDMA特性的需求也日益增强。本文分析了TensorFlow RDMA特性如何与原有的gRPC无缝整合，解析了TensorFlow分布式设计理念以及如何实现RDMA的初始化和数据传输。希望本文可以给使用TensorFlow RDMA的用户带来帮助、给想要推出RDMA特性的框架设计者提供一些启发。







