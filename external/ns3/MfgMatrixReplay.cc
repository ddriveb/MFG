// MFG optional uneven MoE replay adapter for SimAI's pinned ns-3 network.
// Include common.h only in the isolated upstream build; no production hooks.
#include "astra-sim/system/Common.hh"
#include "common.h"
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <tuple>

struct ReplayFlow { unsigned src, dst; uint64_t bytes; };
struct ReplayPhase { string name; vector<ReplayFlow> flows; };
unsigned replay_gpus;
vector<pair<uint64_t,uint64_t>> replay_compute;
vector<ReplayPhase> replay_phases;
size_t replay_phase = 0, replay_pending = 0;
uint64_t replay_received = 0, replay_expected = 0, replay_start_ns = 0;
map<tuple<unsigned,unsigned,uint16_t>, uint64_t> replay_live;
ofstream replay_audit, replay_summary;
bool replay_complete = false;

void BeginReplayPhase();
void ReplayNoop(void*) {}
void ReplaySendFinished(FILE*, Ptr<RdmaQueuePair>) {}

void AfterReplayCompute() {
    BeginReplayPhase();
}

void EndReplayPhase() {
    if(replay_received != replay_expected || !replay_live.empty())
        throw runtime_error("flow conservation failed");
    uint64_t end = Simulator::Now().GetNanoSeconds();
    replay_summary << replay_phases[replay_phase].name << " " << replay_start_ns << " " << end << " " << replay_expected << "\n";
    replay_summary.flush();
    ++replay_phase;
    if(replay_phase == 4) {
        replay_summary << "total_ns " << end << "\n";
        replay_summary.flush();
        replay_complete = true;
        Simulator::Stop();
    } else if(replay_phase == 1 || replay_phase == 3) {
        uint64_t max_ns = 0;
        for(auto t : replay_compute) max_ns = max(max_ns, replay_phase == 1 ? t.first : t.second);
        replay_summary << (replay_phase == 1 ? "expert_forward" : "expert_backward") << " " << end << " " << end + max_ns << " 0\n";
        // All ranks compute concurrently; next global collective starts when
        // all complete. No compute/collective overlap in this frozen replay.
        Simulator::Schedule(NanoSeconds(max_ns), &AfterReplayCompute);
    } else {
        Simulator::ScheduleNow(&BeginReplayPhase);
    }
}

void ReplayReceived(FILE*, Ptr<RdmaQueuePair> q) {
    auto key = make_tuple(ip_to_node_id(q->sip), ip_to_node_id(q->dip), q->sport);
    auto it = replay_live.find(key);
    if(it == replay_live.end() || it->second != q->m_size)
        throw runtime_error("unknown, duplicate or size-mismatched QP completion");
    replay_received += q->m_size;
    replay_audit << replay_phase << " " << get<0>(key) << " " << get<1>(key) << " " << q->m_size << " " << Simulator::Now().GetNanoSeconds() << "\n";
    replay_live.erase(it);
    if(--replay_pending == 0) EndReplayPhase();
}

void BeginReplayPhase() {
    replay_start_ns = Simulator::Now().GetNanoSeconds();
    replay_pending = replay_phases[replay_phase].flows.size();
    replay_expected = replay_received = 0;
    if(replay_pending == 0) { Simulator::Schedule(MicroSeconds(20), &EndReplayPhase); return; }
    // Explicit fixed 20us collective invocation startup from frozen model.
    // Unlike upstream per-QP NCCL latency, paid once per global phase.
    for(auto f : replay_phases[replay_phase].flows) {
        replay_expected += f.bytes;
        uint16_t port = portNumber[f.src][f.dst]++;
        auto key = make_tuple(f.src, f.dst, port);
        if(!replay_live.emplace(key, f.bytes).second) throw runtime_error("duplicate live QP");
        RdmaClientHelper helper(3, serverAddress[f.src], serverAddress[f.dst], port, 100, f.bytes,
            has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(f.src)][n.Get(f.dst)]) : 0,
            global_t == 1 ? maxRtt : pairRtt[f.src][f.dst], &ReplayNoop, nullptr,
            static_cast<int>(replay_phase), f.src, f.dst);
        auto app = helper.Install(n.Get(f.src));
        app.Start(MicroSeconds(20)); // ns3 Application initializes relative to current event
    }
}

void LoadReplay(const string& path) {
    ifstream in(path); string magic;
    in >> magic >> replay_gpus;
    if(!in || magic != "MFG_MATRIX_REPLAY_V1" || replay_gpus == 0 || replay_gpus > 512)
        throw runtime_error("invalid replay header");
    replay_compute.resize(replay_gpus);
    for(auto& t : replay_compute) in >> t.first >> t.second;
    const vector<string> names={"forward_dispatch","forward_combine","backward_combine_grad","backward_dispatch_grad"};
    for(auto name : names) {
        ReplayPhase p; size_t count;
        in >> p.name >> count;
        if(!in || p.name != name || count > replay_gpus*(replay_gpus-1)) throw runtime_error("invalid replay phase");
        set<pair<unsigned,unsigned>> edges;
        for(size_t j=0;j<count;++j) {
            ReplayFlow f; in >> f.src >> f.dst >> f.bytes;
            if(!in || f.src >= replay_gpus || f.dst >= replay_gpus || f.src == f.dst || !f.bytes || !edges.emplace(f.src,f.dst).second)
                throw runtime_error("invalid or duplicate nonzero edge");
            p.flows.push_back(f);
        }
        replay_phases.push_back(p);
    }
    string trailing; if(in >> trailing) throw runtime_error("trailing replay input");
}

int main(int argc, char** argv) {
    try {
        if(argc != 5) throw runtime_error("usage: driver topology config replay output_prefix");
        LoadReplay(argv[3]);
        replay_audit.open(string(argv[4])+".flows.txt");
        replay_summary.open(string(argv[4])+".summary.txt");
        if(!replay_audit || !replay_summary) throw runtime_error("output open failed");
        Time::SetResolution(Time::NS); // upstream common.h RTT/BDP uses GetTimeStep as ns
        if(!ReadConf(argv[1],argv[2])) throw runtime_error("config read failed");
        SetConfig();
        SetupNetwork(&ReplayReceived,&ReplaySendFinished);
        if(node_num - switch_num - nvswitch_num != replay_gpus) throw runtime_error("GPU topology mismatch");
        Simulator::ScheduleNow(&BeginReplayPhase);
        Simulator::Stop(Seconds(2)); // incomplete/deadlocked workloads fail below
        Simulator::Run();
        if(!replay_complete) throw runtime_error("simulation stopped before all flows completed");
        Simulator::Destroy();
        return 0;
    } catch(const exception& e) { cerr << "REPLAY_ERROR " << e.what() << "\n"; return 2; }
}
