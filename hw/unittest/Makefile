all:
	$(MAKE) -C generic_queue
	$(MAKE) -C mem_scheduler
	$(MAKE) -C cache_top
	$(MAKE) -C core_top
	$(MAKE) -C issue_top
	$(MAKE) -C local_mem_top
	$(MAKE) -C mem_unit_top
	$(MAKE) -C tcu_fedp
	$(MAKE) -C cta_dispatcher
	$(MAKE) -C kmu

run:
	$(MAKE) -C generic_queue run
	$(MAKE) -C mem_scheduler run
	$(MAKE) -C cache_top run
	$(MAKE) -C core_top run
	$(MAKE) -C issue_top run
	$(MAKE) -C local_mem_top run
	$(MAKE) -C mem_unit_top run
	$(MAKE) -C tcu_fedp run
	$(MAKE) -C cta_dispatcher run
	$(MAKE) -C kmu run

clean:
	$(MAKE) -C generic_queue clean
	$(MAKE) -C mem_scheduler clean
	$(MAKE) -C cache_top clean
	$(MAKE) -C core_top clean
	$(MAKE) -C issue_top clean
	$(MAKE) -C local_mem_top clean
	$(MAKE) -C mem_unit_top clean
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C cta_dispatcher clean
	$(MAKE) -C kmu clean