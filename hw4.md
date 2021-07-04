# Step1: Run demo
```bash
./ci/blackbox.sh --driver=rtlsim --app=demo --cores=1  --args="-n100" --debug
```
# Step2: 
Open the file `run.log`, and search for load requests.
We can find a following prefetch load request with each original load request. 

For example:
```
        2757: D$0 Rd Req: wid=0, PC=80000544, tmask=0001, addr={0xfffffb30, 0xfffffb30, 0xfffffb30, 0x80001b30}, tag=100000a880, byteen=ffff, rd=14, is_dup=1
        ...
        2759: D$0 Rd Req: wid=0, PC=80000544, tmask=0001, addr={0xfffffb34, 0xfffffb34, 0xfffffb34, 0x80001b34}, tag=100000a881, byteen=ffff, rd=14, is_dup=1


        3789: D$0 Rd Req: wid=0, PC=80000454, tmask=0001, addr={0xfefff404, 0xfefff804, 0xfefffc04, 0xfefffff4}, tag=1000008a83, byteen=ffff, rd=9, is_dup=1
        ...
        3791: D$0 Rd Req: wid=0, PC=80000454, tmask=0001, addr={0xfefff408, 0xfefff808, 0xfefffc08, 0xfefffff8}, tag=1000008a81, byteen=ffff, rd=9, is_dup=1
```
From the `addr`, we can see the prefetch request load exactly original addr+4.