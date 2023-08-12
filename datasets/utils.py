class WfPreInstance():
    def __init__(self) -> None:
        # general info of instance
        self.nr_jobs = 0
        self.nr_operations = 0
        self.nr_machines = 0

        # indexing
        self.ix_jobstage = {}  # ix to jobstage
        self.jobstage_ix = {}  # job stage to ix
        self.job_ix = {}  # job name to ix
        self.ix_job = {}  # ix to job name
        self.ix_operation = {}  # ix to operation name
        self.ix_machines = {}  # ix to machine name
        self.machine_ix = {}  # machine name to ix
        self.job_stages = []  # list of all existing job-stage combos

        # job data
        self.job_deadlines = {}
        self.job_duration = {}
        self.job_release_dates = {}
        self.job_nr_operations = {}
        self.job_first_operation = {}
        self.job_quantity = {}

        # operation data
        self.ope_tool_consumption = {}
        self.ope_pallet = {}
        self.ope_night_setup = {}
        self.ope_night_operation = {}
        self.ope_material = {}
        self.ope_machines = {}
        self.ope_tools = {}
        self.ope_spacers = {}
        self.ope_clamp_orientation = {}
        self.ope_clamp_tops = {}

        # operation-operation data
        self.preconstr = {}
        self.sdst = {}

        # operation-machine data
        self.duration = {}

        # others
        self.config = 'comp_dataset'
        self.timestamp_start = 0  # zero point of schedule
        self.schedule = {'allocation': [], "order": []}

    def __str__(self):
        """Return a string representation of the instance."""
        return "nr_jobs: " + str(self.nr_jobs) + " nr_operations: " + str(self.nr_operations) + " nr_machines: " + str(self.nr_machines)


class WfInstance():
    def __init__(self, *args, **kwargs) -> None:
        # general info of instance
        self.nr_jobs = 0
        self.nr_operations = 0
        self.nr_machines = 0

        self.instance_name = ''

        # indexing
        self.ix_jobstage = {}  # ix to jobstage
        self.jobstage_ix = {}  # job stage to ix
        self.job_ix = {}  # job name to ix
        self.ix_job = {}  # ix to job name
        self.ix_operation = {}  # ix to operation name
        self.ix_machines = {}  # ix to machine name
        self.machine_ix = {}  # machine name to ix
        self.job_stages = []  # list of all existing job-stage combos

        # job data
        self.job_deadlines = []
        self.job_duration = []
        self.job_release_dates = []
        self.job_nr_operations = []
        self.job_first_operation = []
        self.job_quantity = []

        # operation data
        self.ope_tool_consumption = []
        self.ope_pallet = []
        self.ope_night_setup = []
        self.ope_night_operation = []
        self.ope_material = []
        self.ope_machines = []
        self.ope_tools = []
        self.ope_spacers = []
        self.ope_clamp_orientation = []
        self.ope_clamp_tops = []

        # operation-operation data
        self.preconstr = []
        self.sdst = []
        self.ope_tool_similarities = []

        # operation-machine data
        self.duration = []

        # others
        self.config = 'comp_dataset'
        self.timestamp_start = 0  # zero point of schedule
        self.schedule = {'allocation': [], "order": []}
        self.machine_schedule = {}
        self.hv_ref = (0, 0)

        for i in args:
            if isinstance(i[0], WfInstance):
                self.__dict__.update(i[0].__dict__)

    def __str__(self):
        """Return a string representation of the instance."""
        return "num_job: " + str(self.nr_jobs) + " num_mas: " + str(self.nr_machines) + " num_opes: " + str(self.nr_operations)


class WfInstanceIndv(WfInstance):
    def __new__(self, *args, **kwargs):
        return super().__new__(self)

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            WfInstance.__init__(self, args[0])
        else:
            WfInstance.__init__(self, args)

        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


if __name__ == '__main__':
    test = WfInstance()
    test.schedule = 'hi'
    test2 = WfInstanceIndv(test)
    print(test2.schedule)
