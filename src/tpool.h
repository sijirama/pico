#pragma once

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct PicoTPoolJob {
    void (*function)(void* arg);
    void* argument;
    struct PicoTPoolJob* next;
};

struct PicoTPool {
    struct PicoTPoolJob* job_head;
    struct PicoTPoolJob* job_tail;
    pthread_t* threads;

    pthread_mutex_t mutex;
    pthread_cond_t work_ready;  // work_cond signals the threads that there is work to be processed
    pthread_cond_t work_done;   // working_cond signals when there are no threads processing.

    size_t active_jobs;  // working_cnt to know how many threads are actively processing work.
    size_t thread_cnt;   // how many threads are alive
    size_t threads_created;

    bool stop;
};

/*
 pico_tpool_create(8)

  for i in 0..7:
      pthread_create(..., pico_tpool_worker, tp)

  Now you have 8 threads all running this same function:

  pico_tpool_worker(tp)

  Each worker loops forever:

  lock pool mutex
  if no jobs:
      sleep on condition variable
  if job exists:
      pop one job from queue
  unlock pool mutex
  run job->function(job->argument)
  lock again
  mark job done
  repeat

  When you call:

  pico_tpool_add_work(tp, some_func, some_arg);

  it adds a job to the queue:

  tp->job_head -> job1 -> job2 -> job3

  Then it signals/broadcasts:

  pthread_cond_broadcast(&tp->work_ready);

  That wakes sleeping workers. One worker grabs one job. Another worker may grab another job. That
 is how work gets executed.

  The array is separate:

  tp->threads[i] = thread;

  That just saves the thread handle so later destroy can do:

  pthread_join(tp->threads[i], NULL);

  So mentally split them:

  worker execution:
      pthread_create starts pico_tpool_worker(tp)

  work dispatch:
      jobs go into tp->job_head / tp->job_tail
      workers pop jobs from that queue

  thread ownership:
      tp->threads[] stores handles only for cleanup

  The worker uses the shared pool pointer tp, not the threads[] array.

 */

static inline void pico_tpool_destroy(struct PicoTPool* tp);

// Work will need to be pulled from the queue at some point to be processed. Since the queue is a
// linked list this handles not only pulling an object from the list but also maintaining the list
// work_first and work_last references for us.
static inline struct PicoTPoolJob* pico_tpool_get_job(struct PicoTPool* tp) {
    struct PicoTPoolJob* work;

    if(tp == NULL) {
        fprintf(stderr, "PicoThreadPoolError: cannot get job from NULL thread pool\n");
        return NULL;
    }

    work = tp->job_head;
    if(work == NULL) {
        return NULL;
    }

    if(work->next == NULL) {
        tp->job_head = NULL;
        tp->job_tail = NULL;
    } else {
        tp->job_head = work->next;
    }

    return work;
}

static inline void tpool_work_destroy(struct PicoTPoolJob* work) {
    if(work == NULL)
        return;
    free(work);
}

static inline void* pico_tpool_worker(void* arg) {
    struct PicoTPool* tp = arg;
    struct PicoTPoolJob* work;

    while(1) {
        if(pthread_mutex_lock(&(tp->mutex)) != 0) {
            fprintf(stderr, "PicoThreadPoolError: worker failed to lock mutex\n");
            return NULL;
        }

        while(tp->job_head == NULL && !tp->stop) {
            if(pthread_cond_wait(&(tp->work_ready), &(tp->mutex)) != 0) {
                fprintf(stderr, "PicoThreadPoolError: worker failed while waiting for work\n");
                pthread_mutex_unlock(&(tp->mutex));
                return NULL;
            }
        }

        if(tp->stop)
            break;

        work = pico_tpool_get_job(tp);
        tp->active_jobs++;

        if(pthread_mutex_unlock(&(tp->mutex)) != 0) {
            fprintf(stderr, "PicoThreadPoolError: worker failed to unlock mutex before job\n");
            return NULL;
        }

        if(work != NULL) {
            work->function(work->argument);
            tpool_work_destroy(work);
        }
        if(pthread_mutex_lock(&(tp->mutex)) != 0) {
            fprintf(stderr, "PicoThreadPoolError: worker failed to lock mutex after job\n");
            return NULL;
        }

        tp->active_jobs--;
        if(!tp->stop && tp->active_jobs == 0 && tp->job_head == NULL)
            if(pthread_cond_signal(&(tp->work_done)) != 0)
                fprintf(stderr, "PicoThreadPoolError: worker failed to signal work completion\n");
        if(pthread_mutex_unlock(&(tp->mutex)) != 0) {
            fprintf(stderr, "PicoThreadPoolError: worker failed to unlock mutex after job\n");
            return NULL;
        }
    }

    tp->thread_cnt--;
    if(pthread_cond_signal(&(tp->work_done)) != 0)
        fprintf(stderr, "PicoThreadPoolError: worker failed to signal shutdown completion\n");
    if(pthread_mutex_unlock(&(tp->mutex)) != 0)
        fprintf(stderr, "PicoThreadPoolError: worker failed to unlock mutex during shutdown\n");

    return NULL;
}

static inline struct PicoTPoolJob* pico_tpool_work_create(void (*func)(void* arg), void* arg) {
    struct PicoTPoolJob* work;

    if(func == NULL) {
        fprintf(stderr, "PicoThreadPoolError: cannot create work with NULL function\n");
        return NULL;
    }

    work = malloc(sizeof(*work));
    if(work == NULL) {
        fprintf(stderr, "PicoThreadPoolError: failed to allocate work item\n");
        return NULL;
    }
    work->function = func;
    work->argument = arg;
    work->next = NULL;
    return work;
}

static inline bool pico_tpool_add_work(struct PicoTPool* tm, void (*function)(void* arg),
                                       void* arg) {
    struct PicoTPoolJob* work;

    if(tm == NULL) {
        fprintf(stderr, "PicoThreadPoolError: cannot add work to NULL thread pool\n");
        return false;
    }

    work = pico_tpool_work_create(function, arg);
    if(work == NULL)
        return false;

    if(pthread_mutex_lock(&(tm->mutex)) != 0) {
        fprintf(stderr, "PicoThreadPoolError: failed to lock queue while adding work\n");
        tpool_work_destroy(work);
        return false;
    }

    if(tm->stop) {
        fprintf(stderr, "PicoThreadPoolError: cannot add work to stopping thread pool\n");
        pthread_mutex_unlock(&(tm->mutex));
        tpool_work_destroy(work);
        return false;
    }

    if(tm->job_head == NULL) {
        tm->job_head = work;
        tm->job_tail = tm->job_head;
    } else {
        tm->job_tail->next = work;
        tm->job_tail = work;
    }

    if(pthread_cond_signal(&(tm->work_ready)) != 0)
        fprintf(stderr, "PicoThreadPoolError: failed to signal available work\n");
    if(pthread_mutex_unlock(&(tm->mutex)) != 0)
        fprintf(stderr, "PicoThreadPoolError: failed to unlock queue after adding work\n");

    return true;
}

static inline struct PicoTPool* pico_tpool_create(size_t num_threads) {
    struct PicoTPool* tm;
    pthread_t thread;
    size_t i;

    if(num_threads == 0)
        num_threads = 2;

    tm = calloc(1, sizeof(*tm));
    if(tm == NULL) {
        fprintf(stderr, "PicoThreadPoolError: failed to allocate thread pool\n");
        return NULL;
    }

    if(pthread_mutex_init(&(tm->mutex), NULL) != 0) {
        fprintf(stderr, "PicoThreadPoolError: failed to initialize mutex\n");
        free(tm);
        return NULL;
    }
    if(pthread_cond_init(&(tm->work_ready), NULL) != 0) {
        fprintf(stderr, "PicoThreadPoolError: failed to initialize work_ready condition\n");
        pthread_mutex_destroy(&(tm->mutex));
        free(tm);
        return NULL;
    }
    if(pthread_cond_init(&(tm->work_done), NULL) != 0) {
        fprintf(stderr, "PicoThreadPoolError: failed to initialize work_done condition\n");
        pthread_cond_destroy(&(tm->work_ready));
        pthread_mutex_destroy(&(tm->mutex));
        free(tm);
        return NULL;
    }

    tm->job_head = NULL;
    tm->job_tail = NULL;
    tm->threads = calloc(num_threads, sizeof(pthread_t));
    if(tm->threads == NULL) {
        fprintf(stderr, "PicoThreadPoolError: failed to allocate worker handles\n");
        pthread_cond_destroy(&(tm->work_done));
        pthread_cond_destroy(&(tm->work_ready));
        pthread_mutex_destroy(&(tm->mutex));
        free(tm);
        return NULL;
    }

    for(i = 0; i < num_threads; i++) {
        if(pthread_create(&thread, NULL, pico_tpool_worker, tm) != 0) {
            fprintf(stderr, "PicoThreadPoolError: failed to create worker thread\n");
            pico_tpool_destroy(tm);
            return NULL;
        }

        tm->threads[i] = thread;
        tm->thread_cnt++;
        tm->threads_created++;
    }

    return tm;
}

static inline void pico_tpool_wait(struct PicoTPool* tp) {
    if(tp == NULL)
        return;

    if(pthread_mutex_lock(&(tp->mutex)) != 0) {
        fprintf(stderr, "PicoThreadPoolError: failed to lock while waiting\n");
        return;
    }
    while(1) {
        if(tp->job_head != NULL || (!tp->stop && tp->active_jobs != 0) ||
           (tp->stop && tp->thread_cnt != 0)) {
            if(pthread_cond_wait(&(tp->work_done), &(tp->mutex)) != 0) {
                fprintf(stderr, "PicoThreadPoolError: failed while waiting for work completion\n");
                break;
            }
        } else {
            break;
        }
    }
    if(pthread_mutex_unlock(&(tp->mutex)) != 0)
        fprintf(stderr, "PicoThreadPoolError: failed to unlock after waiting\n");
}

static inline void pico_tpool_destroy(struct PicoTPool* tp) {
    struct PicoTPoolJob* work;
    struct PicoTPoolJob* work2;
    size_t i;

    if(tp == NULL)
        return;

    if(pthread_mutex_lock(&(tp->mutex)) != 0) {
        fprintf(stderr, "PicoThreadPoolError: failed to lock during destroy\n");
        return;
    }
    work = tp->job_head;
    while(work != NULL) {
        work2 = work->next;
        tpool_work_destroy(work);
        work = work2;
    }
    tp->job_head = NULL;
    tp->job_tail = NULL;
    tp->stop = true;
    if(pthread_cond_broadcast(&(tp->work_ready)) != 0)
        fprintf(stderr, "PicoThreadPoolError: failed to wake workers during destroy\n");
    if(pthread_mutex_unlock(&(tp->mutex)) != 0)
        fprintf(stderr, "PicoThreadPoolError: failed to unlock during destroy\n");

    for(i = 0; i < tp->threads_created; i++) {
        if(pthread_join(tp->threads[i], NULL) != 0)
            fprintf(stderr, "PicoThreadPoolError: failed to join worker thread\n");
    }

    if(pthread_mutex_destroy(&(tp->mutex)) != 0)
        fprintf(stderr, "PicoThreadPoolError: failed to destroy mutex\n");
    if(pthread_cond_destroy(&(tp->work_ready)) != 0)
        fprintf(stderr, "PicoThreadPoolError: failed to destroy work_ready condition\n");
    if(pthread_cond_destroy(&(tp->work_done)) != 0)
        fprintf(stderr, "PicoThreadPoolError: failed to destroy work_done condition\n");

    free(tp->threads);
    free(tp);
}
