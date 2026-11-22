import csv, sys, collections
short = {'gpu__time_duration.sum':'t_us','dram__bytes_read.sum':'dramR','dram__bytes_write.sum':'dramW','lts__t_bytes.sum':'L2B','lts__t_sectors_op_read.sum':'L2rd','lts__t_sectors_op_write.sum':'L2wr','lts__t_sector_op_read_hit_rate.pct':'rdHit%','lts__t_sector_op_write_hit_rate.pct':'wrHit%','l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum':'ldReq','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum':'ldSec','l1tex__t_requests_pipe_lsu_mem_global_op_st.sum':'stReq','l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum':'stSec','sm__throughput.avg.pct_of_peak_sustained_elapsed':'SM%','lts__throughput.avg.pct_of_peak_sustained_elapsed':'L2%','dram__throughput.avg.pct_of_peak_sustained_elapsed':'DRAM%','l1tex__throughput.avg.pct_of_peak_sustained_elapsed':'L1%','smsp__inst_executed.sum':'inst','sm__warps_active.avg.pct_of_peak_sustained_active':'occ%','l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum':'bankc','smsp__issue_active.avg.pct_of_peak_sustained_active':'issue%','launch__occupancy_limit_shared_mem':'blkSM','launch__shared_mem_per_block_dynamic':'smemB','lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum':'rdMiss','lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum':'wrMiss'}
for f in sys.argv[1:]:
    rows = collections.OrderedDict()
    txt = open(f).read().splitlines()
    i = next(k for k,l in enumerate(txt) if l.startswith('"ID"'))
    for r in csv.DictReader(txt[i:]):
        k = (r['ID'], r['Kernel Name'], r['Grid Size'])
        rows.setdefault(k, {})[r['Metric Name']] = r['Metric Value'].replace(',', '')
    print('==', f)
    for k, m in rows.items():
        def g(n):
            v = m.get(n); 
            try: return float(v)
            except: return float('nan')
        t = g('gpu__time_duration.sum') / 1e3
        s = f"  #{k[0]} grid{k[2]} t {t:.2f}us dram R/W {g('dram__bytes_read.sum')/1e9:.3f}/{g('dram__bytes_write.sum')/1e9:.3f} GB  L2 rd/wr {g('lts__t_sectors_op_read.sum')*32/1e9:.3f}/{g('lts__t_sectors_op_write.sum')*32/1e9:.3f} GB hit {g('lts__t_sector_op_read_hit_rate.pct'):.0f}/{g('lts__t_sector_op_write_hit_rate.pct'):.0f}%"
        s += f"  ld sec/req {g('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')/max(1,g('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum')):.1f} st {g('l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum')/max(1,g('l1tex__t_requests_pipe_lsu_mem_global_op_st.sum')):.1f}"
        s += f"  SM {g('sm__throughput.avg.pct_of_peak_sustained_elapsed'):.0f}% L1 {g('l1tex__throughput.avg.pct_of_peak_sustained_elapsed'):.0f}% L2 {g('lts__throughput.avg.pct_of_peak_sustained_elapsed'):.0f}% DRAM {g('dram__throughput.avg.pct_of_peak_sustained_elapsed'):.0f}% issue {g('smsp__issue_active.avg.pct_of_peak_sustained_active'):.0f}% occ {g('sm__warps_active.avg.pct_of_peak_sustained_active'):.0f}% blk/SM {g('launch__occupancy_limit_shared_mem'):.0f} smem {g('launch__shared_mem_per_block_dynamic')/1e3:.1f}KB bankc {g('l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum')/1e6:.2f}M inst {g('smsp__inst_executed.sum')/1e6:.1f}M"
        print(s)
