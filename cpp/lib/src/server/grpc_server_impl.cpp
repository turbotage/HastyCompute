module;

#include <grpcpp/grpcpp.h>
#include "hasty_service.grpc.pb.h"
#include "hasty_service.pb.h"

module hasty_server_mod;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_generic_value_mod;
import hasty_threading_mod;

// ---------------------------------------------------------------------------
// UUID helpers
// ---------------------------------------------------------------------------

namespace {

std::string uuid_key(const hasty::Uuid& proto) {
    return proto.value();
}

hasty::Uuid to_uuid_proto(const std::array<std::uint8_t, 16>& uuid) {
    hasty::Uuid proto;
    proto.set_value(std::string(reinterpret_cast<const char*>(uuid.data()), 16));
    return proto;
}

} // anonymous namespace


class HastyServiceImpl final : public hasty::HastyService::Service {
public:
    HastyServiceImpl(hasty::GenericValueBank& bank, hasty::CommandRegistry& registry)
        : _bank(bank), _registry(registry) {}

    grpc::Status PushValue(
        grpc::ServerContext*,
        grpc::ServerReader<hasty::DataChunk>* reader,
        hasty::Uuid* response) override
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
            *response = to_uuid_proto(fut.get());
            return grpc::Status::OK;
        } catch (const std::exception& e) {
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    grpc::Status FetchValue(
        grpc::ServerContext*,
        const hasty::FetchRequest* request,
        grpc::ServerWriter<hasty::DataChunk>* writer) override
    {
        auto uuid = uuid_key(request->id());
        if (!_bank.contains(uuid)) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND, "UUID not found in bank");
        }

        hasty::GenericValue value;
        auto slice_info = request->slice_info();
        if (slice_info.empty()) {
            value = std::move(_bank.fetch_value(uuid));
        } else {
            try {
                value = std::move(_bank.fetch_value(uuid, slice_info));
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
                    hasty::DataChunk dc;
                    dc.set_data(data.data(), data.size());
                    writer->Write(dc);
                }
            } catch (const std::runtime_error&) {
                // timeout — loop again, is_finished() exits when done
            }
        }

        fut.get();
        return grpc::Status::OK;
    }

    grpc::Status WriteValue(
        grpc::ServerContext*,
        grpc::ServerReader<hasty::WriteMessage>* reader,
        hasty::WriteAck* response) override
    {
        hasty::WriteMessage msg;
        if (!reader->Read(&msg) || !msg.has_header()) {
            response->set_success(false);
            response->set_error_msg("First message must be WriteHeader");
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

    grpc::Status DeleteValue(
        grpc::ServerContext*,
        const hasty::Uuid* request,
        hasty::WriteAck* response) override
    {
        if (_bank.delete_value(uuid_key(*request))) {
            response->set_success(true);
        } else {
            response->set_success(false);
            response->set_error_msg("UUID not found in bank");
        }
        return grpc::Status::OK;
    }

    grpc::Status Execute(
        grpc::ServerContext*,
        const hasty::ExecuteCommand* request,
        hasty::ExecuteAck* response) override
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
            const auto& opt = _bank.fetch_value(uuid);
            inputs.push_back(opt);
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

private:
    hasty::GenericValueBank&  _bank;
    hasty::CommandRegistry&   _registry;
};

// ---------------------------------------------------------------------------
// ServerHandle::Impl + method definitions
// ---------------------------------------------------------------------------

namespace hasty {

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

hasty::GrpcServerHandle hasty::start_grpc_server(
    hasty::GenericValueBank& bank,
    hasty::CommandRegistry& registry,
    const std::string& address)
{
    hasty::GrpcServerHandle handle;
    handle._impl = std::make_unique<GrpcServerHandle::Impl>();
    handle._impl->service = std::make_unique<HastyServiceImpl>(bank, registry);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(handle._impl->service.get());
    builder.SetMaxReceiveMessageSize(-1);
    builder.SetMaxSendMessageSize(-1);
    builder.AddChannelArgument("grpc.http2.initial_window_size",
                            128 * 1024 * 1024);   // already there
    /*
    builder.AddChannelArgument("grpc.http2.initial_connection_window_size",
    128 * 1024 * 1024);   // connection-level window (push direction)
    builder.AddChannelArgument("grpc.http2.bdp_probe", 0);  // no BDP pings; windows are fixed
    */
        
    handle._impl->server  = builder.BuildAndStart();
    handle._impl->address = address;

    return std::move(handle);
}


}