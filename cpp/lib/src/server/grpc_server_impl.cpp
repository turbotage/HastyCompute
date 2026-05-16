module;

#include <grpcpp/grpcpp.h>
#include <absl/log/log_sink.h>
#include <absl/log/log_sink_registry.h>
#include "hasty_service.grpc.pb.h"
#include "hasty_service.pb.h"

module hasty_server_mod;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_generic_value_mod;
import hasty_threading_mod;

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

enum class BankQueryType : std::int32_t {
    LIST_UUIDS = 0,
};

namespace {

std::string uuid_key(const hasty::Uuid& proto) {
    return proto.value();
}

hasty::Uuid to_uuid_proto(const std::array<std::uint8_t, 16>& uuid) {
    hasty::Uuid proto;
    proto.set_value(std::string(reinterpret_cast<const char*>(uuid.data()), 16));
    return proto;
}

// ── gRPC internal-log forwarding via absl::LogSink ──────────────────────────
// Modern gRPC uses abseil logging — register a global LogSink to capture it.
// We own both the sink and the OStreamInterface at module scope.

class GrpcLogSink final : public absl::LogSink {
public:
    explicit GrpcLogSink(hasty::OStreamInterface& stream) : _stream(stream) {}

    void Send(const absl::LogEntry& entry) override {
        auto sv = entry.text_message_with_prefix_and_newline();
        _stream << std::string(sv.data(), sv.size());
    }
    void Flush() override {}

private:
    hasty::OStreamInterface& _stream;
};

std::mutex                               s_grpc_log_mtx;
std::unique_ptr<hasty::OStreamInterface> s_grpc_internal_log;
std::unique_ptr<GrpcLogSink>             s_grpc_log_sink;

} // anonymous namespace

// Check whether a TCP port is already in use on the specified host:port.
static bool is_port_in_use(const std::string& host, const std::string& port) {
    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) {
        // Could not resolve — conservatively assume not in use.
        return false;
    }
    bool in_use = false;
    for (struct addrinfo* rp = res; rp != nullptr; rp = rp->ai_next) {
        int s = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (s == -1) continue;
        int opt = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        if (::bind(s, rp->ai_addr, rp->ai_addrlen) == -1) {
            if (errno == EADDRINUSE) {
                in_use = true;
                ::close(s);
                break;
            }
            ::close(s);
            continue;
        }
        // Successfully bound — close and report not in use
        ::close(s);
        in_use = false;
        break;
    }
    freeaddrinfo(res);
    return in_use;
}


class HastyServiceImpl final : public hasty::HastyService::Service {
public:
    HastyServiceImpl(hasty::server::GenericValueBank& bank, hasty::server::CommandRegistry& registry)
        : _bank(bank), _registry(registry) {}

    grpc::Status Push(
        grpc::ServerContext*,
        grpc::ServerReader<hasty::DataChunk>* reader,
        hasty::PushResponse* response) override
    {
        hasty::threadsafe_stream stream;

        auto fut = std::async(std::launch::async, [&]() -> std::array<std::uint8_t, 16> {
            return _bank.push_value(hasty::GenericValue::deserialize(stream));
        });

        hasty::DataChunk chunk;
        while (reader->Read(&chunk)) {
            const auto& d = chunk.data();
            stream.write(std::vector<std::uint8_t>(d.begin(), d.end()));
        }
        stream.set_finished();

        try {
            *response->mutable_id() = to_uuid_proto(fut.get());
            response->set_success(true);
            return grpc::Status::OK;
        } catch (const std::exception& e) {
            response->set_success(false);
            response->set_msg(e.what());
            return grpc::Status::OK;
        }
    }

    grpc::Status Read(
        grpc::ServerContext*,
        const hasty::ReadRequest* request,
        grpc::ServerWriter<hasty::ReadResponse>* writer) override
    {
        auto uuid = uuid_key(request->id());
        if (!_bank.contains(uuid)) {
            hasty::ReadResponse err;
            err.mutable_header()->set_success(false);
            err.mutable_header()->set_msg("UUID not found in bank");
            writer->Write(err);
            return grpc::Status::OK;
        }

        hasty::ReadResponse hdr;
        hdr.mutable_header()->set_success(true);
        writer->Write(hdr);

        hasty::GenericValue value;
        const auto& slice_info = request->slice_info();
        if (slice_info.empty()) {
            value = _bank.fetch_value(uuid);
        } else {
            try {
                value = _bank.fetch_value(uuid, slice_info);
            } catch (const std::exception& e) {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
            }
        }

        hasty::threadsafe_stream stream;

        auto fut = std::async(std::launch::async, [&stream, value = std::move(value)]() mutable {
            hasty::GenericValue::serialize(std::move(value), stream);
            stream.set_finished();
        });

        while (!stream.is_finished()) {
            try {
                auto data = stream.read_chunk_blocking(std::chrono::milliseconds(100));
                if (!data.empty()) {
                    hasty::ReadResponse resp;
                    resp.mutable_data()->set_data(data.data(), data.size());
                    writer->Write(resp);
                }
            } catch (const std::runtime_error&) {
                // timeout — loop again
            }
        }

        fut.get();
        return grpc::Status::OK;
    }

    grpc::Status Write(
        grpc::ServerContext*,
        grpc::ServerReader<hasty::WriteRequest>* reader,
        hasty::WriteResponse* response) override
    {
        hasty::WriteRequest msg;
        if (!reader->Read(&msg) || !msg.has_header()) {
            response->set_success(false);
            response->set_error_msg("First message must be WriteRequestHeader");
            return grpc::Status::OK;
        }

        auto key = uuid_key(msg.header().id());
        if (!_bank.contains(key)) {
            response->set_success(false);
            response->set_error_msg("UUID not found in bank");
            return grpc::Status::OK;
        }

        const std::string slice_info = msg.header().slice_info();
        hasty::threadsafe_stream stream;

        auto fut = std::async(std::launch::async, [&stream, key, slice_info, this]() {
            if (slice_info.empty()) {
                _bank.write_value(key, hasty::GenericValue::deserialize(stream));
            } else {
                _bank.write_value(key, slice_info, hasty::GenericValue::deserialize(stream));
            }
        });

        while (reader->Read(&msg)) {
            if (msg.has_data()) {
                const auto& d = msg.data().data();
                stream.write(std::vector<std::uint8_t>(d.begin(), d.end()));
            }
        }
        stream.set_finished();

        try {
            fut.get();
            response->set_success(true);
        } catch (const std::exception& e) {
            response->set_success(false);
            response->set_error_msg(e.what());
        }
        return grpc::Status::OK;
    }

    grpc::Status ExecuteCommand(
        grpc::ServerContext*,
        const hasty::ExecuteCommandRequest* request,
        hasty::ExecuteCommandResponse* response) override
    {
        std::vector<hasty::GenericValue> inputs;
        inputs.reserve(request->input_ids_size());

        for (const auto& uuid_proto : request->input_ids()) {
            auto uuid = uuid_key(uuid_proto);
            if (!_bank.contains(uuid)) {
                response->set_success(false);
                response->set_error_msg("Input UUID not found in bank: " + uuid);
                return grpc::Status::OK;
            }
            inputs.push_back(_bank.fetch_value(uuid));
        }

        try {
            auto [msg, outputs] = _registry.execute(
                request->function_id(), request->options(), std::move(inputs));
            for (auto& out : outputs)
                *response->add_output_ids() = to_uuid_proto(_bank.push_value(std::move(out)));
            response->set_msg(msg);
            response->set_success(true);
        } catch (const std::exception& e) {
            response->set_success(false);
            response->set_error_msg(e.what());
        }
        return grpc::Status::OK;
    }

    grpc::Status BankQuery(
        grpc::ServerContext*,
        const hasty::BankQueryRequest* request,
        hasty::BankQueryResponse* response) override
    {
        try {
            switch (static_cast<BankQueryType>(request->query_type())) {
            case BankQueryType::LIST_UUIDS: {
                auto keys = _bank.list_keys();
                for (const auto& key : keys) {
                    hasty::Uuid* uuid = response->add_ids_in_bank();
                    uuid->set_value(key);
                }
                response->set_success(true);
                break;
            }
            default:
                response->set_success(false);
                response->set_error_msg("Unknown query_type: " +
                    std::to_string(request->query_type()));
                break;
            }
        } catch (const std::exception& e) {
            response->set_success(false);
            response->set_error_msg(e.what());
        }
        return grpc::Status::OK;
    }

    grpc::Status Delete(
        grpc::ServerContext*,
        const hasty::Uuid* request,
        hasty::DeleteResponse* response) override
    {
        if (_bank.delete_value(uuid_key(*request))) {
            response->set_success(true);
        } else {
            response->set_success(false);
            response->set_msg("UUID not found in bank");
        }
        return grpc::Status::OK;
    }

    grpc::Status ReadMetadata(
        grpc::ServerContext*,
        const hasty::MetadataReadRequest* request,
        hasty::MetadataReadResponse* response) override
    {
        auto key = uuid_key(request->id());
        auto opt = _bank.read_metadata(key);
        if (opt) {
            response->set_success(true);
            response->set_metadata_string(*opt);
        } else {
            response->set_success(false);
            response->set_msg("No metadata for UUID");
        }
        return grpc::Status::OK;
    }

    grpc::Status WriteMetadata(
        grpc::ServerContext*,
        const hasty::MetadataWriteRequest* request,
        hasty::MetadataWriteResponse* response) override
    {
        auto key = uuid_key(request->id());
        try {
            _bank.write_metadata(key, request->metadata_string());
            response->set_success(true);
        } catch (const std::exception& e) {
            response->set_success(false);
            response->set_msg(e.what());
        }
        return grpc::Status::OK;
    }

    grpc::Status DeleteMetadata(
        grpc::ServerContext*,
        const hasty::MetadataDeleteRequest* request,
        hasty::MetadataDeleteResponse* response) override
    {
        if (_bank.delete_metadata(uuid_key(request->id()))) {
            response->set_success(true);
        } else {
            response->set_success(false);
            response->set_msg("No metadata for UUID");
        }
        return grpc::Status::OK;
    }

private:
    hasty::server::GenericValueBank&  _bank;
    hasty::server::CommandRegistry&   _registry;
};

// ---------------------------------------------------------------------------
// ServerHandle::Impl + method definitions
// ---------------------------------------------------------------------------

namespace hasty {
namespace server {

struct GrpcServerHandle::Impl {
    std::unique_ptr<HastyServiceImpl> service;
    std::unique_ptr<grpc::Server>     server;
    std::string                       address;
};

GrpcServerHandle::~GrpcServerHandle() = default;
GrpcServerHandle::GrpcServerHandle(GrpcServerHandle&&) noexcept = default;
GrpcServerHandle& GrpcServerHandle::operator=(GrpcServerHandle&&) noexcept = default;

void GrpcServerHandle::wait()               { _impl->server->Wait(); }
void GrpcServerHandle::shutdown()           { _impl->server->Shutdown(); }
const std::string& GrpcServerHandle::address() const { return _impl->address; }

// ---------------------------------------------------------------------------
// start_server
// ---------------------------------------------------------------------------

GrpcServerHandle start_grpc_server(
    GenericValueBank& bank,
    CommandRegistry& registry,
    const std::string& address,
    UPtr<OStreamInterface> log_stream,
    UPtr<OStreamInterface> internal_log_stream
    )
{
    // Register absl log sink so gRPC internal logs go to internal_log_stream.
    {
        std::lock_guard<std::mutex> lk(s_grpc_log_mtx);
        if (s_grpc_log_sink) absl::RemoveLogSink(s_grpc_log_sink.get());
        s_grpc_internal_log = std::move(internal_log_stream);
        if (s_grpc_internal_log) {
            s_grpc_log_sink = std::make_unique<GrpcLogSink>(*s_grpc_internal_log);
            absl::AddLogSink(s_grpc_log_sink.get());
        } else {
            s_grpc_log_sink.reset();
        }
    }

    GrpcServerHandle handle;
    handle._impl = std::make_unique<GrpcServerHandle::Impl>();
    handle._impl->service = std::make_unique<HastyServiceImpl>(bank, registry);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(handle._impl->service.get());
    builder.SetMaxReceiveMessageSize(-1);
    builder.SetMaxSendMessageSize(-1);
    builder.AddChannelArgument("grpc.http2.initial_window_size",
                               128 * 1024 * 1024);

    if (log_stream)
        *log_stream << "Starting gRPC server on " + address + "...\n";
    // Check whether the address:port is already in use and fail early with a
    // clear message to the provided log stream. Address is expected as
    // host:port (e.g. "0.0.0.0:50051").
    auto pos = address.rfind(':');
    if (pos != std::string::npos) {
        std::string host = address.substr(0, pos);
        std::string port = address.substr(pos + 1);
        if (is_port_in_use(host.empty() ? "0.0.0.0" : host, port)) {
            if (log_stream) {
                *log_stream << "Port " + port + " appears to be in use; aborting gRPC server start.\n";
            }
            throw std::runtime_error("gRPC port " + port + " is already in use");
        }
    }

    handle._impl->server  = builder.BuildAndStart();
    handle._impl->address = address;

    if (log_stream)
        *log_stream << "gRPC server started on " + address + "\n";

    return handle;
}


}
}
