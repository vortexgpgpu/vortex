all:
	$(MAKE) -C generic_queue
	$(MAKE) -C mem_scheduler
	$(MAKE) -C cache
	$(MAKE) -C core
	$(MAKE) -C issue
	$(MAKE) -C local_mem
	$(MAKE) -C mem_unit
	$(MAKE) -C tcu_fedp
	$(MAKE) -C cta_dispatcher
	$(MAKE) -C kmu
	$(MAKE) -C dxa_core
	$(MAKE) -C tcu_unit

run:
	$(MAKE) -C generic_queue run
	$(MAKE) -C mem_scheduler run
	$(MAKE) -C cache run
	$(MAKE) -C core run
	$(MAKE) -C issue run
	$(MAKE) -C local_mem run
	$(MAKE) -C mem_unit run
	$(MAKE) -C tcu_fedp run
	$(MAKE) -C cta_dispatcher run
	$(MAKE) -C kmu run
	$(MAKE) -C dxa_core run
	$(MAKE) -C tcu_unit run

clean:
	$(MAKE) -C generic_queue clean
	$(MAKE) -C mem_scheduler clean
	$(MAKE) -C cache clean
	$(MAKE) -C core clean
	$(MAKE) -C issue clean
	$(MAKE) -C local_mem clean
	$(MAKE) -C mem_unit clean
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C cta_dispatcher clean
	$(MAKE) -C kmu clean
	$(MAKE) -C dxa_core clean
	$(MAKE) -C tcu_unit clean
